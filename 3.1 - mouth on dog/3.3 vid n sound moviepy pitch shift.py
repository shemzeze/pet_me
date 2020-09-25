from moviepy.editor import *
import numpy as np



clip = VideoFileClip("dog_out_smooth.mp4")
clip2 = VideoFileClip("Mouth.mp4")
clip3 = VideoFileClip("Dog.mp4")
# clip2 = clip2.set_fps(30.0)
w, h = clip.size
# print(clip.fps, clip2.fps)
aclip = clip.audio
aclip2 = clip2.audio
aclip3 = clip3.audio
aclip2 = aclip2.to_soundarray(fps=44100, nbytes=2)

def speedx(sound_array, factor):
    # """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]


aclip2 = speedx(aclip2, -12)


def stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( len(sound_array)/f + window_size)

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] += hanning_window*a2_rephased

    result = ((2**(16-4)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')


aclip2 = stretch(aclip2, f=0.5, window_size=2.0, h=2.0)

def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)


aclip2 = pitchshift(aclip2, 12)
# clip.set_duration(7.0)
# aclip2.set_duration(aclip.duration)
# sound_both = concatenate_audioclips([aclip, aclip2])
sound_both2 = CompositeAudioClip([aclip3, aclip2])
clip.audio = sound_both2
clip.write_videofile("new.mp4")
# sound_both2.nchannels = max([2])
# sound_both2.write_audiofile("both_sound.mp3")
# print()
