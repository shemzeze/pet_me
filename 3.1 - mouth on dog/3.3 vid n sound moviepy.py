from moviepy.editor import *

clip = VideoFileClip("Dog small.mp4")
clip2 = VideoFileClip("Mouth.mp4")
w, h = clip.size
sound_both = CompositeAudioClip([clip.volumex(1.2), clip2])
sound_both.nchannels = max([1])
sound_both.write_audiofile("both_sound.mp3",  nchannels=1)
# print()
