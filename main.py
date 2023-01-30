import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy

from scipy.io import wavfile

class_names = list(map(str.strip, open("class_names.txt", "r").readlines()))

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

wav_file_name = 'speech_whistling2.wav'
# wav_file_name = 'miaow_16k.wav'
# wav_file_name = 'meow_jane_2.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

waveform = np.float32(wav_data / tf.int16.max)[:15600]


def tflite_classify(path: str, input_data):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    output = interpreter.get_output_details()[0]  # Model has single output.
    input = interpreter.get_input_details()[0]  # Model has single input.
    interpreter.set_tensor(input['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output['index'])


def ov_classify(path: str, input_data):
    from openvino.runtime import Core
    core = Core()
    model = core.compile_model(path, "CPU")
    return model(input_data)


if __name__ == "__main__":
    path = "lite-model_yamnet_classification_tflite_1.tflite"
    result_lite = tflite_classify(path, waveform)
    infered_class = class_names[result_lite.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class} per tflite. {result_lite.max()*100}% sure')

    result_ov = list(ov_classify(path, waveform).values())[0]
    infered_class = class_names[result_ov.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class} per ov. {result_ov.max()*100}% sure')