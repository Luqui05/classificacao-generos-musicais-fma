"""Extrai atributos de um MP3 inteiro e consulta cinco pipelines confiáveis."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from src.inference import infer_mp3

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path)
    parser.add_argument('--models',type=Path,required=True,help='Diretório confiável com manifest.json e cinco joblib')
    parser.add_argument('--ffmpeg',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'artifacts/inference'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args()
    try:
        result=infer_mp3(args.audio,args.models,args.ffmpeg,args.output)
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        parser.exit(2,f'Erro: {error}\n')
    for name,genre in result['predictions'].items():
        print(f'{name}: {genre}')
    print(f'Registro: {args.output.resolve()}')

if __name__=='__main__':
    main()
