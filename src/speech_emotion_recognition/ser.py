import click
import tomli
from pathlib import Path
from .data_parser import parse_all
from .feature_extraction import preprocess_all
from .train import train_emotion_hmms, train_vocal_hmms
from .eval import evaluate_emotion_hmms, evaluate_vocal_hmms

@click.group()
@click.option(
    "--config", "-c",
    default="config.toml",
    type=click.Path(exists=True),
    help="Path to TOML configuration file"
)
@click.pass_context
def cli(ctx, config):
    """Speech‑Emotion Recognition CLI."""
    with open(config, "rb") as f:
        ctx.obj = tomli.load(f)

@cli.command()
@click.pass_context
def preprocess(ctx):
    """Parse metadata and pre‑process all raw audio into `preproc_dir`."""
    cfg = ctx.obj
    preprocess_all(
        raw_base=Path(cfg["paths"]["raw_data_dir"]),
        preproc_base=Path(cfg["paths"]["preproc_dir"]),
        sr=cfg["mfcc"]["sr"]
    )
    parse_all(
        raw_base=Path(cfg["paths"]["preproc_dir"]),
        cleaned_csv=Path(cfg["paths"]["cleaned_csv"])
    )

@cli.command()
@click.option(
    "--mode", "-m",
    type=click.Choice(["emotion", "vocal"], case_sensitive=False),
    default="emotion",
    help="Which HMMs to train: 'emotion' or 'vocal'"
)
@click.pass_context
def train(ctx, mode):
    """Train HMMs for either emotion recognition or vocal‑channel classification."""
    cfg = ctx.obj
    if mode.lower() == "emotion":
        train_emotion_hmms(cfg)
    else:
        train_vocal_hmms(cfg)

@cli.command(name="eval")
@click.option(
    "--mode", "-m",
    type=click.Choice(["emotion", "vocal"], case_sensitive=False),
    default="emotion",
    help="Which HMMs to evaluate: 'emotion' or 'vocal'"
)

@click.pass_context
def evaluate(ctx, mode):
    """Evaluate HMMs on held‑out data for emotion or vocal‑channel."""
    cfg = ctx.obj
    if mode.lower() == "emotion":
        evaluate_emotion_hmms(cfg)
    else:
        evaluate_vocal_hmms(cfg)

if __name__ == "__main__":
    cli()
