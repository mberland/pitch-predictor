from collections import Counter
from os.path import dirname, exists
from os import path, makedirs
from sklearn.model_selection import train_test_split
from time import strftime, localtime
import bz2
import copy
import numpy as np
import pickle
import random


from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

start_time_str = strftime("%Y%m%d-%H%M",localtime())

TESTING_FLAG = True

if TESTING_FLAG:
    print("WARNING: TESTING MODE! Set TESTING_FLAG = False to run full model.", flush=True)
    num_epochs = 1
else:
    num_epochs = 50 # You can increase this until loss plateaus.

def encode(data,vocab):
    encoded = to_categorical(vocab.index(data),num_classes=len(vocab))
    return encoded

def decode(datum,vocab):
    return vocab[np.argmax(datum)]

directories = ["data","models","ckpt"]
for dir in directories:
    if not exists(dir):
        try:
            makedirs(dir)
        except:
            print("ERROR: Could not create and/or access directory: {}".format(dir))
            raise

games_filename = lambda y : "data/games-p4-" + str(y) + ".pkl.bz2"

all_games = {}
data_years = [2016,2017,2018,2019]
random.shuffle(data_years)
    
valid_years = []
for year in data_years:
    if path.exists(games_filename(year)):
        print("File (" + games_filename(year) + ") found. Loading...",end="",flush=True)
        FILE_games = bz2.open(games_filename(year),"rb")
        years_games = pickle.load(FILE_games)
        print("Updating...",end="",flush=True)
        all_games.update(years_games)
        FILE_games.close()
        print("DONE")
        valid_years.append(str(year))
        if TESTING_FLAG:
            break
    else:
        print("File (" + games_filename(year) + ") NOT found.",flush=True)
print("YEARS LOADED: {}".format(" ".join(valid_years)))



def get_playerDB_by_game(c_game):
    players = c_game['gameData']['players']
    player_db = {}
    for elt in players.values():
        player_db[elt['id']] = elt['fullName']
    return player_db

def get_teamDB_by_game(c_game):
    team_db = {}
    for side in ['home','away']:
        team_name = c_game['liveData']['boxscore']['teams'][side]['team']['name']
        for player in c_game['liveData']['boxscore']['teams'][side]['players'].values():
            team_db[player['person']['id']] = team_name
    return team_db

field_out = ['balk', 'batter_interference', 'catcher_interf', 'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home',  'double_play', 'fan_interference', 'field_error', 'field_out', 'fielders_choice', 'fielders_choice_out', 'force_out', 'game_advisory', 'grounded_into_double_play', 'hit_by_pitch',  'other_out', 'passed_ball', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home', 'pickoff_error_1b', 'runner_double_play','stolen_base_2b','triple_play', 'wild_pitch','sac_bunt', 'sac_bunt_double_play', 'sac_fly', 'sac_fly_double_play']
walk = ['intent_walk','walk']
strikeout = ['strikeout', 'strikeout_double_play']
hit = ['home_run', 'single', 'double', 'triple']
def play_result_category(play_result):
    if play_result in walk:
        return 'walk'
    if play_result in strikeout:
        return 'strikeout'
    if play_result in hit:
        return 'hit'
    return 'field_out'


class XPlayEvent:
    def __init__(self, play, team_db, player_db):
        self.max_pitches = len(play['pitchIndex'])
        self.start_time = play['about']['startTime']
        self.play_result = play_result_category(play['result']['eventType']) # string from vocab
        self.inning = play['about']['inning'] # num
        self.away_bat = play['about']['isTopInning'] # bool
        self.away_score = play['result']['awayScore'] # num
        self.home_score = play['result']['homeScore'] # num
        self.pitcher_id = play['matchup']['pitcher']['id']
        self.batter_id = play['matchup']['batter']['id']
        self.men_on = (0 if "Empty" == play['matchup']['splits']['menOnBase'] else 1)
        self.pitcher_team = team_db[self.pitcher_id]
        self.batter_team = team_db[self.batter_id]
        self.pitcher = player_db[self.pitcher_id]
        self.batter = player_db[self.batter_id]
        if (self.away_bat):
            self.batter_score = self.away_score
            self.pitcher_score = self.home_score
        else:
            self.pitcher_score = self.away_score
            self.batter_score = self.home_score

class XPitchEvent:
    def __init__(self, play, playevent, pitch_num, pitch_seq):
        self.pe = playevent
        self.balls = play['playEvents'][pitch_num]['count']['balls']
        self.strikes = play['playEvents'][pitch_num]['count']['strikes']
        self.outs = play['count']['outs']
        self.pitch_num = pitch_num
        self.pitch = play['playEvents'][pitch_num]['details']['type']['code']
        self.prior_seq = pitch_seq

def arr2pe(pitch_array):
    raise NotImplementedError
    return False


def pitch_events_by_game(c_game):
    output = []
    team_db = get_teamDB_by_game(c_game)
    player_db = get_playerDB_by_game(c_game)
    number_of_plays = len(c_game['liveData']['plays']['allPlays'])
    for play_number in range(number_of_plays):
        play = c_game['liveData']['plays']['allPlays'][play_number]
        if ('event' in play['result']):
            play_event = XPlayEvent(play, team_db, player_db)
            filtered_pitch_indices = list(filter(lambda i : play['playEvents'][i]["isPitch"] and "type" in play['playEvents'][i]['details'], range(play_event.max_pitches)))
            pitch_seq = []
            for i in filtered_pitch_indices:
                pitch_event = XPitchEvent(play,play_event,i,copy.deepcopy(pitch_seq))
                pitch_seq.append(pitch_event.pitch)
                output.append(pitch_event)
    return output

print("Making xevent_list...",end="",flush=True)
xevent_list = []
success = [xevent_list.extend(pitch_events_by_game(t_game)) for t_game in all_games.values()]
xplay_list = {x.pe.start_time: x.pe for x in xevent_list}
xplay_list = list(xplay_list.values())
print("DONE.",flush=True)


all_players = set(map(lambda x: x.pe.pitcher, xevent_list))
all_players.update(map(lambda x: x.pe.batter, xevent_list))
all_players = sorted(all_players)
print("ALL PLAYERS[",len(all_players),"] ( e.g., ",random.sample(all_players,3),")")

all_pitchtypes = sorted(set(map(lambda x: x.pitch, xevent_list)))
print("ALL PITCHTYPES[",len(all_pitchtypes),"]:", all_pitchtypes)

all_outcometypes = sorted(set(map(lambda x: x.pe.play_result, xevent_list)))
print("ALL RESULTS[",len(all_outcometypes),"]:", all_outcometypes)


print("Vectorizing players...",flush=True)
all_player_pitchcounts = {player: [] for player in all_players}
all_player_outcomecounts = {player: [] for player in all_players}
for elt in xevent_list:
    all_player_pitchcounts[elt.pe.pitcher].append(elt.pitch)
    all_player_pitchcounts[elt.pe.batter].append(elt.pitch)
for elt in xplay_list:
    all_player_outcomecounts[elt.pitcher].append(elt.play_result)
    all_player_outcomecounts[elt.batter].append(elt.play_result)

all_pitch_counters = {x: Counter(all_player_pitchcounts[x]) for x in all_player_pitchcounts.keys()}
all_outcome_counters = {x: Counter(all_player_outcomecounts[x]) for x in all_player_outcomecounts.keys()}

all_vectors = {}
for player in all_pitch_counters:
    all_vectors[player] = [all_pitch_counters[player][pitchtype] for pitchtype in all_pitchtypes]
    all_vectors[player].extend([all_outcome_counters[player][outcometype] for outcometype in all_outcometypes])

random_player = random.choice(all_players)
print(",".join(all_pitchtypes + all_outcometypes))
print(",".join(map(lambda x : "{:2d}".format(x), all_vectors[random_player])))
print("DONE.",flush=True)

def pe2arr(pitch_event):
    if len(pitch_event.prior_seq) > 0:
        last_pitch_id = all_pitchtypes.index(pitch_event.prior_seq[-1])
    else:
        last_pitch_id = -1
    out = [pitch_event.balls, pitch_event.strikes,
            pitch_event.outs, pitch_event.pe.batter_score,
            pitch_event.pe.pitcher_score, pitch_event.pe.inning, last_pitch_id]
    out.extend(all_vectors[pitch_event.pe.batter])
    out.extend(all_vectors[pitch_event.pe.pitcher])
    return np.array(out)

def pe2result(pitch_event):
    return encode(pitch_event.pitch,all_pitchtypes)


print(pe2arr(random.choice(xevent_list)),flush=True)

print("Splitting for evaluation...",end="",flush=True)
xevent_list, eval_xevent_list = train_test_split(xevent_list,test_size=0.1)
print("DONE.",flush=True)


print("Making input/output arrays...",end="",flush=True)
input_array = np.array(list(map(lambda x : pe2arr(x),xevent_list))) 
output_array = np.array(list(map(lambda x : pe2result(x),xevent_list)))
print("DONE.",flush=True)

print("Building model...",end="",flush=True)
model = Sequential()
model.add(Dense(1000, input_shape=input_array[0].shape))
model.add(BatchNormalization())
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(len(output_array[0]), activation='softmax'))
optimizer = SGD()
print("DONE.")
print(model.summary())

model.compile(loss='kullback_leibler_divergence', metrics=['accuracy'], optimizer=optimizer)

checkpoint_path = "./ckpt/pp-" + start_time_str + "-{epoch:04d}.ckpt"
checkpoint_dir = dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_path)

# this split is more for just basic sanity checking at checkpoints.
X_train, X_test, y_train, y_test = train_test_split(input_array, output_array, test_size=0.05)

model.fit(X_train, y_train , epochs=num_epochs,
            callbacks=[cp_callback], validation_data=(X_test,y_test))
model_path = "./models/model-" + start_time_str + ".h5" 
model.save(model_path)
loss, acc = model.evaluate(X_test,y_test, verbose=2)
print("Model accuracy: {:5.2f}%".format(100*acc))

thresholds = [0.05,0.1,0.2]
place_threshold = 3
success_above_threshold = Counter()
success_above_rank = 0
exact_success = 0
validation_sample_size = len(eval_xevent_list)

print("Evaluating against {} samples...".format(validation_sample_size),end="",flush=True)
for x in eval_xevent_list:
    prediction = model.predict(np.array([pe2arr(x)]))
    prediction_list = prediction.tolist()[0]
    index_pitch = all_pitchtypes.index(x.pitch)
    index_predmax = np.argmax(prediction_list)
    nth_rank = sorted(prediction_list,reverse=True)[place_threshold]
    for threshold in thresholds:
        if (index_pitch in filter(lambda p : prediction_list[p] > threshold, range(len(prediction_list)))):
            success_above_threshold[threshold] += 1
    if (index_pitch in filter(lambda p : prediction_list[p] > nth_rank, range(len(prediction_list)))):
        success_above_rank += 1
    if (index_pitch == index_predmax):
        exact_success += 1
print("DONE.")

for threshold in thresholds:
    percent_above_threshold = (100.0*success_above_threshold[threshold])/validation_sample_size
    print("{:2.2f}% above {:2.0f}% predicted chance".format(percent_above_threshold,(100*threshold)))
percent_above_rank = (100.0*success_above_rank)/validation_sample_size
print("{:2.2f}% at rank {} or better".format(percent_above_rank,place_threshold))
percent_exact_correct = (100.0*exact_success)/validation_sample_size
print("{:2.2f}% exact success".format(percent_exact_correct,place_threshold))
