import regelum as rg
from src.callback import ROSScenarioStepLogger

@rg.main(config_name="main", 
         config_path="presets",
         callbacks=[
             ROSScenarioStepLogger
         ])
def launch(cfg):
    scenario = ~cfg.scenario

    scenario.run()


if __name__ == "__main__":
    launch()
