import statsapi
import pickle
from random import shuffle
from os.path import exists
from os import makedirs
from bz2 import open

directories = ["data","models","ckpt"]
for dir in directories:
    if not exists(dir):
        try:
            makedirs(dir)
        except:
            print("ERROR: Could not create and/or access directory: {}".format(dir))
            raise

all_games = {}
data_years = [2016,2017,2018,2019]
shuffle(data_years)

schedule_filename = lambda y : "data/schedule-p4-" + str(y) + ".pkl.bz2"
games_filename = lambda y : "data/games-p4-" + str(y) + ".pkl.bz2"

schedule = {}
all_games = {}

for year in data_years:
    print(year,": ",schedule_filename(year),games_filename(year))
    if exists(schedule_filename(year)):
        print("{} exists. Loading...".format(schedule_filename(year)),end="")
        FILE_schedule = open(schedule_filename(year),"rb")
        schedule = pickle.load(FILE_schedule)
        print("DONE.")
    else:
        print("{} does not exist. Downloading...".format(schedule_filename(year)),end="")
        FILE_schedule = open(schedule_filename(year),"wb")
        schedule = statsapi.schedule(start_date='01/01/'+str(year),end_date='12/31/'+str(year))
        pickle.dump(schedule,FILE_schedule,protocol=4)
        print("DONE.")

    FILE_schedule.close()
    
    game_ids = [x['game_id'] for x in schedule]

    if exists(games_filename(year)):
        print("File (" + games_filename(year) + ") found. No update.",flush=True)
    else:
        print("File (" + games_filename(year) + ") NOT found. Downloading...",end="",flush=True)
        counter = 0
        for game_id in game_ids:
            game = statsapi.get("game",{"gamePk":game_id})
            all_games[game_id] = game
            counter += 1
            print('.', end='')
            if (10 == counter % 100):
                print(str(counter),flush=True)
                FILE_games = open(games_filename(year),"wb")
                pickle.dump(all_games,FILE_games,protocol=4)
                FILE_games.close()
        FILE_games = open(games_filename(year),"wb")
        pickle.dump(all_games,FILE_games,protocol=4)
        FILE_games.close()
        print("DONE.")

