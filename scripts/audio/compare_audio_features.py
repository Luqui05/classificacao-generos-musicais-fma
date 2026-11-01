"""Prepara uma amostra FMA ou compara seus MP3s com os atributos oficiais."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.audio_validation import PROJECT_ROOT, prepare_sample, run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "compare"])
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/fma_small_features.csv")
    parser.add_argument("--official-features", type=Path,
                        default=PROJECT_ROOT / "data/fma_metadata/features.csv")
    parser.add_argument("--sample", type=Path, default=PROJECT_ROOT / "validation/audio_features/sample")
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT / "data/fma_small")
    parser.add_argument("--output", type=Path, help="Pasta nova; pastas existentes são rejeitadas.")
    parser.add_argument("--per-genre", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = args.output or args.sample
            manifest = prepare_sample(args.dataset, args.official_features, output,
                                      per_genre=args.per_genre, seed=args.seed)
            print(manifest.to_string(index=False))
            print(f"\nAmostra preparada em: {output}")
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or PROJECT_ROOT / "validation/audio_features/runs" / stamp
        code = run_comparison(args.sample, args.official_features, args.audio_root, output,
                              rtol=args.rtol, atol=args.atol)
        print(f"Relatório: {output}")
        print({0: "Valores dentro das tolerâncias diagnósticas.",
               1: "Diferenças acima das tolerâncias diagnósticas.",
               2: "Comparação incompleta: veja os arquivos tracks.csv e run.json."}[code])
        return code
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(2, f"Erro: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
