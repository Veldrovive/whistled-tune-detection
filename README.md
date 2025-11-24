

BasicPitch is a more modern pitch detection system that uses a neural network. I don't know whether it makes sense to use this on a PI. It also doesn't seem like it supports streaming.
https://huggingface.co/spotify/basic-pitch

Perhaps look at if this repo has it working well
https://github.com/exirmee/pitchdetector

Interesting discussion
https://forum.juce.com/t/lowest-latency-real-time-pitch-detection/51741/26


Perhaps these are all too rigid. I want to match against a library of arbitrary pitch sequences.
A simple threshold above 0dB separates whistling nicely. Humming is always bad.
I could set a dynamic threshold based on the stats of the current spectrogram as well so that when it is quiet we can still pick.
Then we do some filtering on the signal to pick out edges which correspond to notes starting or ending. Something like a sobel filter in only the x direction.

New idea:
Use mel spectrogram to better separate pitches we care about.
Use some simple adaptive noise suppression and then threshold to separate out new noises.
Blur and x sobel to extract pitch start and end.
Convert to binary with threshold > 0 to extract onset only.
Erode until pitches one note apart separate visually.
Extract centroids of each blob to get exact pitch onset and frequency.
Compare deltas with dictionary with some probabilistic model to get codebook probabilities.
Codebook could include distributions over time between notes and delta amount to get probabilities.
Whistle it multiple times to get the distributions.

We basically do an MLE at that point. On each frame or at a set interval (or on new note detection), we extract our note sequences then for each code in the codebook we try to attribute each note in the code to a note in the detection and select the one with the maximum likelyhood as our candidate for that code. Then if it exceeds a threshold we say we have a detection. This can potentially work even in noisy areas because as long as the code is detected as separated notes it will be detected. If we have adaptive noise removal that is filtering our regular speaking it could even work when things are not well separated.

When we use label to convert the blobs into note detections, we can also get a distribution over possible onset times and pitches. This is less important when notes are well separated and distinct, but if they blend together then we might have a single connected component representing multiple voiced pitches. In this case we want some heuristic by which we say two pitches have been voiced. Probably when there is enough mass in a place separated from the original detected pitch by one note.

Audio lib changes:
Construct maps from x and y index to time and frequency.
Change to mel spectrogram
Make better
overlapping chunks?


Look into scipy.signal.find_peaks for a more adaptive way to find candidates for whistles
Look into skimage.measure.regionprops to better filter once countours have been found

For activation:
Use MFCC or whatever it is called to try to classify when a whistle occurs.