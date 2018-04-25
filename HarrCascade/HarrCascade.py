import os

text=''
for filename in os.listdir('positive_images'):
    text = text+'positive_images/'+filename+  '  1  0  0  28  28\n'

print(text)
with open('info.dat', 'w') as file:
    file.write(text)