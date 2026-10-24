import hydra

from src.datasets.code_preparation import prepare


@hydra.main(version_base=None, config_path="src/configs", config_name="prepare_code")
def main(config):
    prepare(config)


if __name__ == "__main__":
    main()
