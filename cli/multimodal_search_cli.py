from lib import multimodal_search as ms
import argparse


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search")
    subparsers = parser.add_subparsers(dest="command", help="Additional commands")
    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding")
    verify_image_embedding_parser.add_argument("img_path", type=str, help="Path to image")
    image_search_parser = subparsers.add_parser("image_search")
    image_search_parser.add_argument("img_path", type=str, help="Path to image")
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            ms.verify_image_embedding(args.img_path)
        case "image_search":
            results = ms.image_search(args.img_path)
            for i, (score, document) in enumerate(results, 1):
                title = document.get("title", "")
                description = document.get("description", "")[:100]
                print(f"{i}. {title} (similarity: {score:0.3f})")
                print(f"\t{description}")
                print()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
