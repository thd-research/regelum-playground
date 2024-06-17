import regelum as rg


@rg.main(config_name="main_ros", config_path="presets")
def launch(cfg):
    scenario = ~cfg.scenario

    scenario.run()


if __name__ == "__main__":
    launch()
