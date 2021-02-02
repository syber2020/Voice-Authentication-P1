#!/usr/bin/env python
# coding: utf-8

# In[27]:


from pydub import AudioSegment
import math

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total_sec = math.ceil(self.get_duration())
        for i in range(0, total_sec, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_sec - sec_per_split:
                print('All splited successfully')


# In[28]:


audio = SplitWavAudioMubin("Y:\\Masters_Content\\Udemy_Full_Stack_Web_Development\\Flask\\Flask-Bootcamp-master\\Flask-Bootcamp-master\\07-User-Authentication\\01-Flask-Login\\downloaded audio clips","At_Valley_Forge__Champ_Clark_.wav")


# In[29]:


audio.multiple_split(10)


# In[ ]:




