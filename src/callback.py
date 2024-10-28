from regelum.callback import (
    HistoricalDataCallback,
    )


class GZHistoricalDataCallback(HistoricalDataCallback):
    def on_episode_done(self, scenario, episode_number, episodes_total, iteration_number, iterations_total):
        print("Enter HistoricalDataCallback on_episode_done")

        print(self.data)
        if episodes_total == 1:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}"
        else:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}_ep_{str(episode_number).zfill(5)}"
        try:
            print("Dump data")
            self.dump_and_clear_data(identifier)
        except Exception as err:
            print("GZHistoricalDataCallback got Error:", err)
        
        print("Leave HistoricalDataCallback on_episode_done")

