"""Training and testing functionality for the de novo peptide sequencing
model."""

import glob
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import lightning.pytorch as pl
import lightning.pytorch.loggers
import numpy as np
import torch
import torch.utils.data
from depthcharge.tokenizers import PeptideTokenizer
from depthcharge.tokenizers.peptides import MskbPeptideTokenizer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .. import utils
from ..config import Config
from ..data import db_utils, ms_io, psm
from ..denovo.dataloaders import DeNovoDataModule
from ..denovo.evaluate import aa_match_batch, aa_match_metrics
from ..denovo.model import DbSpec2Pep, Spec2Pep

logger = logging.getLogger("casanovo")


class ModelRunner:
    """A class to run Casanovo models.

    Parameters
    ----------
    config : Config object
        The casanovo configuration.
    model_filename : str, optional
        The model filename is required for eval and de novo modes,
        but not for training a model from scratch.
    output_dir : Path | None, optional
        The directory where checkpoint files will be saved. If `None` no
        checkpoint files will be saved and a warning will be triggered.
    output_rootname : str | None, optional
        The root name for checkpoint files (e.g., checkpoints or
        results). If `None` no base name will be used for checkpoint
        files.
    overwrite_ckpt_check: bool, optional
        Whether to check output_dir (if not `None`) for conflicting
        checkpoint files.
    """

    def __init__(
        self,
        config: Config,
        model_filename: Optional[str] = None,
        output_dir: Optional[Path | None] = None,
        output_rootname: Optional[str | None] = None,
        overwrite_ckpt_check: Optional[bool] = True,
    ) -> None:
        """Initialize a ModelRunner."""
        self.config = config
        self.model_filename = model_filename
        self.output_dir = output_dir
        self.output_rootname = output_rootname
        self.overwrite_ckpt_check = overwrite_ckpt_check

        # Initialized later.
        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        if output_dir is None:
            self.callbacks = []
            logger.warning(
                "Checkpoint directory not set in ModelRunner, no "
                "checkpoint files will be saved."
            )
            return

        prefix = f"{output_rootname}." if output_rootname is not None else ""
        curr_filename = prefix + "{epoch}-{step}"
        best_filename = prefix + "best"
        if overwrite_ckpt_check:
            utils.check_dir_file_exists(
                output_dir,
                [
                    f"{curr_filename.format(epoch='*', step='*')}.ckpt",
                    f"{best_filename}.ckpt",
                ],
            )

        # Configure checkpoints.
        self.callbacks = [
            ModelCheckpoint(
                dirpath=output_dir,
                save_on_train_epoch_end=True,
                filename=curr_filename,
                enable_version_counter=False,
            ),
            ModelCheckpoint(
                dirpath=output_dir,
                monitor="valid_CELoss",
                filename=best_filename,
                enable_version_counter=False,
            ),
        ]

    def __enter__(self):
        """Enter the context manager."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup on exit."""
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

    def db_search(
        self,
        peak_path: Iterable[str],
        fasta_path: str,
        results_path: str,
    ) -> None:
        """
        Perform database search with Casanovo.

        Parameters
        ----------
        peak_path : Iterable[str]
            The paths with the MS data files for database search.
        fasta_path : str
            The path with the FASTA file for database search.
        results_path : str
            Sequencing results file path.
        """
        self.writer = ms_io.MztabWriter(results_path)
        self.writer.set_metadata(
            self.config,
            model=str(self.model_filename),
            config_filename=self.config.file,
        )
        self.initialize_trainer(train=True)
        self.initialize_tokenizer()
        self.initialize_model(train=False, db_search=True)
        self.model.out_writer = self.writer
        self.model.protein_database = db_utils.ProteinDatabase(
            fasta_path,
            self.config.enzyme,
            self.config.digestion,
            self.config.missed_cleavages,
            self.config.min_peptide_len,
            self.config.max_peptide_len,
            self.config.max_mods,
            self.config.precursor_mass_tol,
            self.config.isotope_error_range,
            self.config.allowed_fixed_mods,
            self.config.allowed_var_mods,
            self.model.tokenizer,
        )
        test_paths = self._get_input_paths(peak_path, False, "test")
        self.writer.set_ms_run(test_paths)
        self.initialize_data_module(test_paths=test_paths)
        self.loaders.protein_database = self.model.protein_database
        self.loaders.setup(stage="test", annotated=False)
        self.trainer.predict(self.model, self.loaders.db_dataloader())

    def train(
        self,
        train_peak_path: Iterable[str],
        valid_peak_path: Iterable[str],
    ) -> None:
        """
        Train the Casanovo model.

        Parameters
        ----------
        train_peak_path : iterable of str
            The path to the MS data files for training.
        valid_peak_path : iterable of str
            The path to the MS data files for validation.
        """
        self.initialize_trainer(train=True)
        self.initialize_tokenizer()
        self.initialize_model(train=True)

        train_paths = self._get_input_paths(train_peak_path, True, "train")
        valid_paths = self._get_input_paths(valid_peak_path, True, "valid")
        self.initialize_data_module(train_paths, valid_paths)
        self.loaders.setup()

        self.trainer.fit(
            self.model,
            self.loaders.train_dataloader(),
            self.loaders.val_dataloader(),
        )

    def log_metrics(self, test_dataloader: DataLoader) -> None:
        """
        Log peptide precision and amino acid precision.

        Calculate and log peptide precision and amino acid precision
        based off of model predictions and spectrum annotations.

        Parameters
        ----------
        test_dataloader : DataLoader
            Index containing the annotated spectra used to generate
            model predictions.
        """
        pred_seqs, true_seqs, pred_i = [], [], 0

        for batch in test_dataloader:
            for peak_file, scan_id, true_seq in zip(
                batch["peak_file"], batch["scan_id"], batch["seq"]
            ):
                true_seqs.append(true_seq.cpu().detach().numpy())
                if pred_i < len(self.writer.psms) and self.writer.psms[
                    pred_i
                ].spectrum_id == (peak_file, scan_id):
                    pred_tokens = self.model.tokenizer.tokenize(
                        self.writer.psms[pred_i].sequence
                    ).squeeze(0)
                    pred_seqs.append(pred_tokens.cpu().detach().numpy())
                    pred_i += 1
                else:
                    pred_seqs.append(None)

        aa_masses = {
            aa_token: self.model.tokenizer.residues[aa]
            for aa, aa_token in self.model.tokenizer.index.items()
            if aa in self.model.tokenizer.residues
        }
        aa_precision, aa_recall, pep_precision = aa_match_metrics(
            *aa_match_batch(true_seqs, pred_seqs, aa_masses)
        )

        if self.config["top_match"] > 1:
            logger.warning(
                "The behavior for calculating evaluation metrics is undefined "
                "when the 'top_match' configuration option is set to a value "
                "greater than 1."
            )

        logger.info("Peptide Precision: %.2f%%", 100 * pep_precision)
        logger.info("Amino Acid Precision: %.2f%%", 100 * aa_precision)
        logger.info("Amino Acid Recall: %.2f%%", 100 * aa_recall)

    def predict(
        self,
        peak_path: Iterable[str],
        results_path: str,
        evaluate: bool = False,
    ) -> None:
        """
        Predict peptide sequences with a trained Casanovo model.

        Can also evaluate model during prediction if provided with
        annotated peak files.

        Parameters
        ----------
        peak_path : Iterable[str]
            The path with the MS data files for predicting peptide
            sequences.
        results_path : str
            Sequencing results file path.
        evaluate: bool
            whether to run model evaluation in addition to inference
            Note: peak_path must point to annotated MS data files when
            running model evaluation. Files that are not an annotated
            peak file format will be ignored if evaluate is set to true.
        """
        self.writer = ms_io.MztabWriter(results_path)
        self.writer.set_metadata(
            self.config,
            model=str(self.model_filename),
            config_filename=self.config.file,
        )

        self.initialize_trainer(train=False)
        self.initialize_tokenizer()
        self.initialize_model(train=False)
        self.model.out_writer = self.writer

        test_paths = self._get_input_paths(peak_path, False, "test")
        self.writer.set_ms_run(test_paths)
        self.initialize_data_module(test_paths=test_paths)

        try:
            self.loaders.setup(stage="test", annotated=evaluate)
        except (KeyError, OSError) as e:
            if evaluate:
                error_message = (
                    "Error creating annotated spectrum dataloaders. This may "
                    "be the result of having an unannotated peak file present "
                    "in the validation peak file path list."
                )

                logger.error(error_message)
                raise TypeError(error_message) from e

            raise

        predict_dataloader = self.loaders.predict_dataloader()
        self.trainer.predict(self.model, predict_dataloader)

        if evaluate:
            self.log_metrics(predict_dataloader)

    def profile(
        self,
        peak_path: Iterable[str],
        profile_path: str,
    ) -> None:
        """
        First-pass greedy profiling for mass-ladder assembly.

        Runs `Spec2Pep.decode_profiles` over every spectrum and
        serializes the per-position token-probability profiles together
        with precursor and vocabulary metadata into a single ``.npz``.
        This is the input to reference-free overlap assembly; no mzTab
        is produced.

        Parameters
        ----------
        peak_path : Iterable[str]
            The paths with the MS data files to profile.
        profile_path : str
            Output path for the serialized profiles (``.npz``).
        """
        self.initialize_trainer(train=False)
        self.initialize_tokenizer()
        self.initialize_model(train=False)

        test_paths = self._get_input_paths(peak_path, False, "test")
        self.initialize_data_module(test_paths=test_paths)
        self.loaders.setup(stage="test", annotated=False)

        if self.config.accelerator == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        model = self.model.to(device)
        model.eval()

        profiles, lengths = [], []
        peak_files, scan_ids = [], []
        precursor_mz, precursor_charge, precursor_mass = [], [], []

        with torch.no_grad():
            for batch in self.loaders.predict_dataloader():
                mzs, ints, precursors, _ = model._process_batch(batch)
                batch_profiles = model.decode_profiles(
                    mzs.to(device), ints.to(device), precursors.to(device)
                )
                pre = precursors.cpu().numpy()
                files = list(batch["peak_file"])
                scans = list(batch["scan_id"])
                for i, prof in enumerate(batch_profiles):
                    profiles.append(prof.astype(np.float32))
                    lengths.append(prof.shape[0])
                    peak_files.append(str(files[i]))
                    scan = scans[i]
                    scan_ids.append(
                        str(scan.item() if torch.is_tensor(scan) else scan)
                    )
                    precursor_mass.append(float(pre[i, 0]))
                    precursor_charge.append(int(pre[i, 1]))
                    precursor_mz.append(float(pre[i, 2]))

        vocab_tokens = [""] * model.vocab_size
        for tok, idx in model.tokenizer.index.items():
            if 0 <= idx < model.vocab_size:
                vocab_tokens[idx] = tok

        profiles_concat = (
            np.concatenate(profiles, axis=0)
            if profiles
            else np.zeros((0, model.vocab_size), dtype=np.float32)
        )

        np.savez_compressed(
            profile_path,
            profiles_concat=profiles_concat,
            lengths=np.asarray(lengths, dtype=np.int32),
            peak_file=np.asarray(peak_files),
            scan_id=np.asarray(scan_ids),
            precursor_mz=np.asarray(precursor_mz, dtype=np.float64),
            precursor_charge=np.asarray(precursor_charge, dtype=np.int32),
            precursor_mass=np.asarray(precursor_mass, dtype=np.float64),
            token_masses=model.token_masses.cpu().numpy(),
            vocab_tokens=np.asarray(vocab_tokens),
            stop_int=np.int32(model.stop_token),
            decoding_order=("C_to_N" if model.tokenizer.reverse else "N_to_C"),
        )
        logger.info(
            "Wrote %d spectrum profiles to %s", len(lengths), profile_path
        )

    def assemble(
        self,
        profile_paths: Iterable[str],
        consensus_path: str,
        bestpath_fasta_path: Optional[str] = None,
        ppm_tolerance: float = 25.0,
    ) -> None:
        """
        Build a mass-coordinate consensus DAG from first-pass profiles.

        Reads one or more ``.npz`` files produced by :meth:`profile`,
        constructs an incremental overlap-layout-consensus DAG in
        cumulative-mass coordinate, and serializes the layout to
        ``consensus_path``. If ``bestpath_fasta_path`` is given, also
        writes the highest-confidence path through the DAG to a FASTA.

        Pure data-stage helper — no model required.
        """
        from .assembly import (
            build_layout,
            load_profiles,
            bestpath,
            save_consensus,
        )

        paths = list(profile_paths)
        reads, meta = load_profiles(paths)
        logger.info(
            "Assembling %d reads from %d profile file(s)",
            len(reads),
            len(paths),
        )
        if not reads:
            logger.warning("No reads to assemble; writing empty consensus")
            save_consensus(
                build_layout(
                    [],
                    meta.get("vocab", np.array([])),
                    meta.get("token_masses", np.array([])),
                    ppm_tolerance=ppm_tolerance,
                ),
                consensus_path,
            )
            return

        layout = build_layout(
            reads,
            vocab=meta["vocab"],
            token_masses=meta["token_masses"],
            ppm_tolerance=ppm_tolerance,
        )
        save_consensus(layout, consensus_path)
        logger.info(
            "Wrote consensus DAG (%d nodes, %d edges) to %s",
            len(layout.nodes),
            len(layout.edges),
            consensus_path,
        )

        if bestpath_fasta_path is not None:
            seq, scores, _ = bestpath(layout)
            mean_score = float(np.mean(scores)) if scores else 0.0
            with open(bestpath_fasta_path, "w") as fh:
                fh.write(
                    f">consensus_bestpath len={len(seq)} "
                    f"mean_conf={mean_score:.4f} "
                    f"placed={len(layout.read_shifts)} "
                    f"unplaced={len(layout.unplaced)}\n"
                )
                for k in range(0, len(seq), 60):
                    fh.write(seq[k : k + 60] + "\n")
            logger.info(
                "Wrote bestpath FASTA (%d residues, mean conf %.3f) to %s",
                len(seq),
                mean_score,
                bestpath_fasta_path,
            )

    def redecode(
        self,
        peak_path: Iterable[str],
        consensus_path: str,
        results_path: str,
        alpha: float = 0.5,
    ) -> None:
        """
        Second-pass biased decoding seeded by a consensus DAG.

        Runs greedy decoding with a per-step prior drawn from the
        mean-fused outgoing distribution at the current cumulative
        mass in the consensus layout. Mirrors :meth:`predict` in I/O
        shape — emits a casanovo-style mzTab — but uses
        :meth:`Spec2Pep.decode_with_priors` instead of beam search.

        Parameters
        ----------
        peak_path : Iterable[str]
            MS data files to re-decode.
        consensus_path : str
            Path to the consensus ``.npz`` from :meth:`assemble`.
        results_path : str
            Output ``.mztab`` path.
        alpha : float, optional
            Blend weight on the prior (0 = pure model, 1 = pure
            consensus). Defaults to 0.5.
        """
        from .assembly import consensus_lookup, load_consensus

        self.writer = ms_io.MztabWriter(results_path)
        self.writer.set_metadata(
            self.config,
            model=str(self.model_filename),
            config_filename=self.config.file,
        )

        self.initialize_trainer(train=False)
        self.initialize_tokenizer()
        self.initialize_model(train=False)

        test_paths = self._get_input_paths(peak_path, False, "test")
        self.writer.set_ms_run(test_paths)
        self.initialize_data_module(test_paths=test_paths)
        self.loaders.setup(stage="test", annotated=False)

        layout = load_consensus(consensus_path)
        global_lookup = consensus_lookup(layout)
        # Decoder may be C→N; the layout's mass coordinate is N→C, so
        # for reverse decoders we lookup against ``precursor - cum``.
        reverse_decoder = self.model.tokenizer.reverse

        if self.config.accelerator == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        model = self.model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in self.loaders.predict_dataloader():
                mzs, ints, precursors, _ = model._process_batch(batch)
                pre_np = precursors.cpu().numpy()
                priors = []
                for i in range(mzs.shape[0]):
                    pmass = float(pre_np[i, 0])
                    if reverse_decoder:
                        priors.append(
                            lambda cum, pmass=pmass: global_lookup(pmass - cum)
                        )
                    else:
                        priors.append(lambda cum: global_lookup(cum))

                results = model.decode_with_priors(
                    mzs.to(device),
                    ints.to(device),
                    precursors.to(device),
                    priors,
                    alpha=alpha,
                )

                files = list(batch["peak_file"])
                scans = list(batch["scan_id"])
                charges = list(batch["precursor_charge"])
                mzs_pre = list(batch["precursor_mz"])
                for i, (tok_ids, step_probs) in enumerate(results):
                    if not tok_ids:
                        continue
                    seq = self.model.tokenizer.detokenize(
                        torch.tensor([tok_ids])
                    )[0]
                    aa_scores = step_probs.astype(np.float32)
                    if reverse_decoder:
                        aa_scores = aa_scores[::-1]
                    pep_score = (
                        float(np.mean(aa_scores)) if aa_scores.size else 0.0
                    )
                    scan = scans[i]
                    self.writer.psms.append(
                        psm.PepSpecMatch(
                            sequence=seq,
                            spectrum_id=(
                                str(files[i]),
                                str(
                                    scan.item()
                                    if torch.is_tensor(scan)
                                    else scan
                                ),
                            ),
                            peptide_score=pep_score,
                            charge=int(charges[i]),
                            calc_mz=float("nan"),
                            exp_mz=float(
                                mzs_pre[i].item()
                                if torch.is_tensor(mzs_pre[i])
                                else mzs_pre[i]
                            ),
                            aa_scores=aa_scores.tolist(),
                        )
                    )

        self.writer.save()
        logger.info(
            "Wrote re-decoded mzTab (alpha=%.2f) to %s", alpha, results_path
        )

    def initialize_trainer(self, train: bool) -> None:
        """
        Initialize the Pytorch Lightning Trainer.

        Parameters
        ----------
        train : bool
            Determines whether to set the trainer up for model training
            or evaluation / inference.
        """
        trainer_cfg = dict(
            accelerator=self.config.accelerator,
            devices=1,
            enable_checkpointing=False,
            precision=self.config.precision,
            logger=False,
        )

        if train:
            if self.config.devices is None:
                devices = "auto"
            else:
                devices = self.config.devices

            # Configure loggers.
            loggers = []
            if self.config.log_metrics or self.config.tb_summarywriter:
                if not self.output_dir:
                    logger.warning(
                        "Output directory not set in model runner. "
                        "No loss file or Tensorboard will be created."
                    )
                else:
                    csv_log_dir = "csv_logs"
                    tb_log_dir = "tensorboard"

                    if self.config.log_metrics:
                        if self.overwrite_ckpt_check:
                            utils.check_dir_file_exists(
                                self.output_dir, csv_log_dir
                            )

                        loggers.append(
                            lightning.pytorch.loggers.CSVLogger(
                                self.output_dir, version=csv_log_dir, name=None
                            )
                        )

                    if self.config.tb_summarywriter:
                        if self.overwrite_ckpt_check:
                            utils.check_dir_file_exists(
                                self.output_dir, tb_log_dir
                            )

                        loggers.append(
                            lightning.pytorch.loggers.TensorBoardLogger(
                                self.output_dir, version=tb_log_dir, name=None
                            )
                        )

                    if len(loggers) > 0:
                        self.callbacks.append(
                            LearningRateMonitor(
                                log_momentum=True, log_weight_decay=True
                            ),
                        )

            additional_cfg = dict(
                devices=devices,
                val_check_interval=self.config.val_check_interval,
                max_epochs=self.config.max_epochs,
                num_sanity_val_steps=self.config.num_sanity_val_steps,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm,
                callbacks=self.callbacks,
                check_val_every_n_epoch=None,
                enable_checkpointing=True,
                logger=loggers,
                strategy=self._get_strategy(),
            )

            trainer_cfg.update(additional_cfg)

        self.trainer = pl.Trainer(**trainer_cfg)

    def initialize_tokenizer(self) -> None:
        """Initialize the peptide tokenizer."""
        if self.config.massivekb_tokenizer:
            tokenizer_clss = MskbPeptideTokenizer
        else:
            tokenizer_clss = PeptideTokenizer

        missing_aa = list(
            set(self.config.residues) - set(tokenizer_clss.residues)
        )
        if missing_aa:
            logger.warning(
                "Configured residue(s) not in model alphabet: %s",
                ", ".join(missing_aa),
            )

        self.tokenizer = tokenizer_clss(
            residues=self.config.residues,
            replace_isoleucine_with_leucine=self.config.replace_isoleucine_with_leucine,
            reverse=True,
            start_token=None,
            stop_token="$",
        )

    def initialize_model(self, train: bool, db_search: bool = False) -> None:
        """Initialize the Casanovo model.

        Parameters
        ----------
        train : bool
            Determines whether to set the model up for model training or
            evaluation / inference.
        db_search : bool
            Determines whether to use the DB search model subclass.
        """
        try:
            tokenizer = self.tokenizer
        except AttributeError:
            raise RuntimeError(
                "The tokenizer must be initialized prior to the model"
            )

        model_params = dict(
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            top_match=self.config.top_match,
            n_beams=self.config.n_beams,
            n_log=self.config.n_log,
            max_charge=self.config.max_charge,
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            train_label_smoothing=self.config.train_label_smoothing,
            calculate_precision=self.config.calculate_precision,
            out_writer=self.writer,
            tokenizer=tokenizer,
        )

        # Reconfigurable non-architecture related parameters for a
        # loaded model.
        loaded_model_params = dict(
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            max_peptide_len=self.config.max_peptide_len,
            top_match=self.config.top_match,
            n_beams=self.config.n_beams,
            n_log=self.config.n_log,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            train_label_smoothing=self.config.train_label_smoothing,
            calculate_precision=self.config.calculate_precision,
            out_writer=self.writer,
        )

        if self.model_filename is None:
            if db_search:
                logger.error("A model file must be provided for DB search")
                raise ValueError("A model file must be provided for DB search")
            # Train a model from scratch if no model file is provided.
            if train:
                self.model = Spec2Pep(**model_params)
                return
            # Else we're not training, so a model file must be provided.
            else:
                logger.error("A model file must be provided")
                raise ValueError("A model file must be provided")
        # Else a model file is provided (to continue training or for
        # inference).

        if not Path(self.model_filename).exists():
            logger.error(
                "Could not find the model weights at file %s",
                self.model_filename,
            )
            raise FileNotFoundError("Could not find the model weights file")

        # First try loading model details from the weights file,
        # otherwise use the provided configuration.
        device = torch.empty(1).device  # Use the default device.
        model_clss = DbSpec2Pep if db_search else Spec2Pep
        try:
            self.model = model_clss.load_from_checkpoint(
                self.model_filename,
                map_location=device,
                weights_only=False,
                **loaded_model_params,
            )
            # Use tokenizer initialized from config file instead of loaded
            # from checkpoint file.
            self.model.tokenizer = tokenizer
            architecture_params = set(model_params.keys()) - set(
                loaded_model_params.keys()
            )
            for param in architecture_params:
                if model_params[param] != self.model.hparams[param]:
                    warnings.warn(
                        f"Mismatching {param} parameter in "
                        f"model checkpoint ({self.model.hparams[param]}) "
                        f"vs config file ({model_params[param]}); "
                        "using the checkpoint."
                    )
        except RuntimeError:
            # This only doesn't work if the weights are from an older
            # version.
            try:
                self.model = model_clss.load_from_checkpoint(
                    self.model_filename,
                    map_location=device,
                    weights_only=False,
                    **model_params,
                )
                self.model.tokenizer = tokenizer
            except RuntimeError:
                raise RuntimeError(
                    "Weights file incompatible with the current version of "
                    "Casanovo."
                )

    def initialize_data_module(
        self,
        train_paths: Sequence[str] | None = None,
        valid_paths: Sequence[str] | None = None,
        test_paths: Sequence[str] | None = None,
    ) -> None:
        """Initialize the data module.

        Parameters
        ----------
        train_paths : str, optional
            Spectrum paths for model training.
        valid_paths : str, optional
            Spectrum paths for validation.
        test_paths : str, optional
            Spectrum paths for evaluation or inference.
        """
        try:
            n_devices = self.trainer.num_devices
            train_batch_size = self.config.train_batch_size // n_devices
            eval_batch_size = self.config.predict_batch_size // n_devices
        except AttributeError:
            raise RuntimeError(
                "The trainer must be initialized prior to the data module"
            )

        try:
            tokenizer = self.tokenizer
        except AttributeError:
            raise RuntimeError(
                "The tokenizer must be initialized prior to the data module"
            )

        lance_dir = (
            Path(self.tmp_dir.name)
            if self.config.lance_dir is None
            else self.config.lance_dir
        )
        self.loaders = DeNovoDataModule(
            train_paths=train_paths,
            valid_paths=valid_paths,
            test_paths=test_paths,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            min_peaks=self.config.min_peaks,
            max_peaks=self.config.max_peaks,
            min_mz=self.config.min_mz,
            max_mz=self.config.max_mz,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
            max_charge=self.config.max_charge,
            tokenizer=tokenizer,
            shuffle=self.config.shuffle,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            n_workers=self.config.n_workers,
            lance_dir=lance_dir,
        )

    def _get_input_paths(
        self,
        peak_path: Iterable[str],
        annotated: bool,
        mode: str,
    ) -> List[str]:
        """
        Get the spectrum input paths.

        Parameters
        ----------
        peak_path : Iterable[str]
            The peak files/directories to check.
        annotated : bool
            Are the spectra expected to be annotated?
        mode : str
            Either "train", "valid", or "test" to specify the Lance file
            name.

        Returns
        -------
        List[str]
            The input spectrum paths for the specified mode.
        """
        ext = (".mgf", ".lance")
        if not annotated:
            ext += (".mzml", ".mzxml")

        filenames = _get_peak_filenames(peak_path, ext)
        if not filenames:
            error_message = f"Could not find {mode} peak files"
            logger.error(error_message + " from %s", peak_path)
            raise FileNotFoundError(error_message)

        is_lance = any([Path(f).suffix.lower() == ".lance" for f in filenames])
        if is_lance and len(filenames) > 1:
            error_message = f"Multiple {mode} spectrum Lance files specified"
            logger.error(error_message)
            raise ValueError(error_message)

        return filenames

    def _get_strategy(self) -> Union[str, DDPStrategy]:
        """
        Get the strategy for the Trainer.

        The DDP strategy works best when multiple GPUs are used. It can
        work for CPU-only, but definitely fails using MPS (the Apple
        Silicon chip) due to Gloo.

        Returns
        -------
        Union[str, DDPStrategy]
            The strategy parameter for the Trainer.
        """
        if self.config.accelerator in ("cpu", "mps"):
            return "auto"
        elif self.config.devices == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"


def _get_peak_filenames(
    paths: Iterable[str], supported_ext: Iterable[str]
) -> List[str]:
    """
    Get all matching peak file names from the path pattern.

    Performs cross-platform path expansion akin to the Unix shell (glob,
    expand user, expand vars).

    Parameters
    ----------
    paths : Iterable[str]
        The path pattern(s).
    supported_ext : Iterable[str]
        Extensions of supported peak file formats.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    found_files = set()
    for path in paths:
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        for fname in glob.glob(path, recursive=True):
            if Path(fname).suffix.lower() in supported_ext:
                found_files.add(fname)
            else:
                warnings.warn(
                    f"Ignoring unsupported peak file: {fname}", RuntimeWarning
                )

    if len(found_files) == 0:
        warnings.warn(
            f"No supported peak files found under path(s): {list(paths)}",
            RuntimeWarning,
        )

    return sorted(list(found_files))
