from music21 import *
import glob
from tqdm import tqdm
import numpy as np
import random
from tensorflow.keras.layers import LSTM,Dense,Input,Dropout
from tensorflow.keras.models import Sequential,Model,load_model 
from sklearn.model_selection import train_test_split

def read_files(file):
  notes=[]
  notes_to_parse=None
  midi=converter.parse(file)
  instrmt=instrument.partitionByInstrument(midi)

  for part in instrmt.parts:
    if 'Piano' in str(part):
      notes_to_parse=part.recurse()
      for element in notes_to_parse:
        if type(element)==note.Note:
          notes.append(str(element.pitch))
        elif type(element)==chord.Chord:
          notes.append('.'.join(str(n) for n in element.normalOrder))

  return notes

file_path=["schubert"]
all_files=glob.glob('All Midi Files/'+file_path[0]+'/*.mid',recursive=True)

notes_array = np.array([read_files(i) for i in tqdm(all_files,position=0,leave=True)])

notess = sum(notes_array,[]) 
unique_notes = list(set(notess))
print("Unique Notes:",len(unique_notes))

freq=dict(map(lambda x: (x,notess.count(x)),unique_notes))

print("\nFrequency notes")
for i in range(30,100,20):
  print(i,":",len(list(filter(lambda x:x[1]>=i,freq.items()))))

freq_notes=dict(filter(lambda x:x[1]>=50,freq.items()))

new_notes=[[i for i in j if i in freq_notes] for j in notes_array]

ind2note=dict(enumerate(freq_notes))

note2ind=dict(map(reversed,ind2note.items()))

timesteps=50

x=[] ; y=[]

for i in new_notes:
  for j in range(0,len(i)-timesteps):
    inp=i[j:j+timesteps] ; out=i[j+timesteps]

    x.append(list(map(lambda x:note2ind[x],inp)))
    y.append(note2ind[out])

x_new=np.array(x) 
y_new=np.array(y)

x_new = np.reshape(x_new,(len(x_new),timesteps,1))
y_new = np.reshape(y_new,(-1,1))

x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,test_size=0.2,random_state=42)


model = Sequential()
model.add(LSTM(256,return_sequences=True,input_shape=(x_new.shape[1],x_new.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))

model.add(Dense(len(note2ind),activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(
    x_train,y_train,
    batch_size=128,epochs=80, 
    validation_data=(x_test,y_test))

model.save("s2s")

model=load_model("s2s")

index = np.random.randint(0,len(x_test)-1)

music_pattern = x_test[index]

out_pred=[] 

for i in range(200):

   
  music_pattern = music_pattern.reshape(1,len(music_pattern),1)
  
  
  pred_index = np.argmax(model.predict(music_pattern))
  
  out_pred.append(ind2note[pred_index])
  music_pattern = np.append(music_pattern,pred_index)
  
 
  music_pattern = music_pattern[1:]

output_notes = []
for offset,pattern in enumerate(out_pred):
  
  if ('.' in pattern) or pattern.isdigit():
    
    notes_in_chord = pattern.split('.')
    notes = []
    for current_note in notes_in_chord:
        i_curr_note=int(current_note)
         
        new_note = note.Note(i_curr_note)
        new_note.storedInstrument = instrument.Piano()
        notes.append(new_note)
    
    new_chord = chord.Chord(notes)
    new_chord.offset = offset
    output_notes.append(new_chord)
  
  else:

    new_note = note.Note(pattern)
    new_note.offset = offset
    new_note.storedInstrument = instrument.Piano()
    output_notes.append(new_note)
 
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='pred_music.mid')
