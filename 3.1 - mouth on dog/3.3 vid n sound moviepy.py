from moviepy.editor import *

clip = VideoFileClip("PetMe out1.mp4")
clip2 = VideoFileClip("Mouth.mp4")
clip3 = VideoFileClip("Dog.mp4")
clip2 = clip2.set_fps(30.0)
w, h = clip.size
print(clip.fps, clip2.fps)
aclip = clip.audio
aclip2 = clip2.audio
aclip3 = clip3.audio
# clip.set_duration(7.0)
# aclip2.set_duration(aclip.duration)
# sound_both = concatenate_audioclips([aclip, aclip2])
sound_both2 = CompositeAudioClip([aclip3, aclip2])
clip.audio = sound_both2
clip.write_videofile("new.mp4")
# sound_both2.nchannels = max([2])
# sound_both2.write_audiofile("both_sound.mp3")
# print()
