#!/usr/bin/env python3
"""Constrói a pasta de treino do Ultralytics a partir dos ficheiros COCO.

Esta pasta é descartável: apaga e volta a correr.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

PASTA_ANOT = "annotations_only_non_and_pollenbee_fixed_final"


def carregar_coco(raiz: Path) -> tuple[dict[str, list], dict[str, tuple[int, int]]]:
    """Caixas e dimensões por nome de imagem, juntando os três ficheiros.

    Quando um nome aparece em mais do que um ficheiro, fica o primeiro.
    É seguro porque o passo 2 já excluiu os casos em que as versões diferiam.
    """
    caixas: dict[str, list] = {}
    dims: dict[str, tuple[int, int]] = {}
    for split in ("train", "val", "test"):
        ficheiro = raiz / split / PASTA_ANOT / f"pollenbee_{split}.json"
        dados = json.loads(ficheiro.read_text(encoding="utf-8"))
        nome_de = {im["id"]: im["file_name"] for im in dados["images"]}
        acc: dict[str, list] = defaultdict(list)
        for a in dados["annotations"]:
            acc[nome_de[a["image_id"]]].append(a["bbox"])
        for im in dados["images"]:
            caixas.setdefault(im["file_name"], acc.get(im["file_name"], []))
            dims.setdefault(im["file_name"], (im["width"], im["height"]))
    return caixas, dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("../Datasets/PollenbeeDataset"))
    ap.add_argument("--splits", type=Path, default=Path("splits/v1"))
    ap.add_argument("--saida", type=Path, default=Path("data/processed/det_vnpollenbee"))
    args = ap.parse_args()

    caixas, dims = carregar_coco(args.dataset)

    if args.saida.exists():
        shutil.rmtree(args.saida)

    vistos: dict[str, str] = {}
    total, recortadas, descartadas = 0, 0, 0

    for conjunto in ("train", "val"):
        dir_img = args.saida / "images" / conjunto
        dir_lab = args.saida / "labels" / conjunto
        dir_img.mkdir(parents=True)
        dir_lab.mkdir(parents=True)

        caminhos = (args.splits / f"{conjunto}.txt").read_text(
            encoding="utf-8").splitlines()

        for caminho in caminhos:
            origem = args.dataset / caminho
            nome = origem.name
            novo = nome.replace(" ", "_")

            if novo in vistos:
                raise SystemExit(f"colisão: '{nome}' e '{vistos[novo]}' dão '{novo}'")
            vistos[novo] = nome

            largura, altura = dims[nome]
            linhas = []
            for x, y, w, h in caixas[nome]:
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(largura), x + w), min(float(altura), y + h)
                if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                    recortadas += 1
                if x2 <= x1 or y2 <= y1:
                    descartadas += 1
                    continue
                linhas.append(
                    f"0 {(x1 + x2) / 2 / largura:.6f} {(y1 + y2) / 2 / altura:.6f}"
                    f" {(x2 - x1) / largura:.6f} {(y2 - y1) / altura:.6f}"
                )

            shutil.copy2(origem, dir_img / novo)
            (dir_lab / f"{Path(novo).stem}.txt").write_text(
                "\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
            total += len(linhas)

        n_img = len(list(dir_img.glob("*.jpg")))
        n_lab = len(list(dir_lab.glob("*.txt")))
        print(f"{conjunto}: {n_img} imagens, {n_lab} rótulos")
        if n_img != n_lab or n_img != len(caminhos):
            raise SystemExit("ERRO: contagens não batem")

    (args.saida / "dataset.yaml").write_text(
        f"path: {args.saida.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: abelha\n",
        encoding="utf-8",
    )

    print(f"\ncaixas escritas: {total}")
    print(f"caixas recortadas à moldura: {recortadas}")
    print(f"caixas descartadas (degeneradas): {descartadas}")
    print(f"dataset.yaml: {(args.saida / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()