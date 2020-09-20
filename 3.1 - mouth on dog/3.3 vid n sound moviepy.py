from moviepy.editor import *

clip = VideoFileClip("Dog small.mp4")
clip2 = VideoFileClip("Mouth.mp4")
w, h = clip.size
aclip = clip.audio
aclip2 = clip2.audio
aclip.set_duration(aclip2.duration)
# sound_both = concatenate_audioclips([aclip, aclip2])
sound_both2 = CompositeAudioClip([aclip, aclip2])
clip.audio = sound_both2
clip.write_videofile("new.mp4")
# sound_both2.nchannels = max([2])
# sound_both2.write_audiofile("both_sound.mp3")
# print()
