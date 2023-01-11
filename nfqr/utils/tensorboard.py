from functools import cached_property
import pickle
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class EventAccumulatorHook(object):
    def __init__(self,source_path,reload=False) -> None:
        self.source_path = source_path

        self.pickleable_acc = {"scalars":{}}
        self.save_path = (self.source_path.parent/(self.source_path.name + ".hook"))
        
        if not self.save_path.exists() or reload:
            self.pickleable_acc["scalars"].update({tag:tuple(map(np.array,zip(*self.accumulator.Scalars(tag)))) for tag in self.accumulator.Tags()["scalars"]})
            self.save()

        else:
            self.pickleable_acc = pickle.load(open(self.save_path,"rb")) 

    def save(self):
        pickle.dump(self.pickleable_acc,open(self.save_path,"wb"))

    @cached_property
    def accumulator(self):
        return EventAccumulator(str(self.source_path)).Reload()

    def Scalars(self,tag):

        if tag not in self.pickleable_acc["scalars"]:
            self.pickleable_acc["scalars"][tag] = self.accumulator.Scalars(tag)
        
        return self.pickleable_acc["scalars"][tag]