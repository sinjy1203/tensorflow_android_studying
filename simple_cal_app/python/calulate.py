### tensorflow 1 버전
# import tensorflow.compat.v1 as tf
# from pathlib import Path
# tf.disable_v2_behavior()
#
# # 이번 파일에서 공통으로 사용하는 함수.
# # 컨버터 생성해서 파일로 저장. 다시 말해 모바일에서 사용할 수 있는 형태로 변환해서 저장.
# def model_common(inputs, outputs, model_path):
#     # 텐서플로 API만을 사용해서 저장할 수 있음을 보여준다.
#     # 4가지 방법 중에서 가장 기본.
#     model_path = model_dir / model_path
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         # input_tensors: List of input tensors. Type and shape are computed using get_shape() and dtype.
#         # output_tensors: List of output tensors (only .name is used from this).
#         converter = tf.lite.TFLiteConverter.from_session(sess,
#                                                          input_tensors=inputs,
#                                                          output_tensors=outputs)
#         # 세션에 들어있는 모든 연산, 즉 모델 전체를 변환
#         # 반환값은 TFLite 형식의 Flatbuffer 또는 Graphviz 그래프
#         flat_data = converter.convert()
#
#         # 텍스트가 아니기 때문에 바이너리 형태로 저장. w(write), b(binary)
#         with open(model_path, 'wb') as f:
#             f.write(flat_data)
#
#
# # 입력 1개, 출력 1개
# def simple_model_1(model_path):
#     # 에러. 반드시 shape을 지정해야 함.
#     # x = tf.placeholder(tf.int32)
#
#     # 안드로이드에서 전달한 입력과 출력 변수가 플레이스 홀더와 연동
#     x = tf.placeholder(tf.int32, shape=[1])
#     out = x * 5
#
#     model_common([x], [out], model_path)
#
#     # 에러. 반드시 [] 형태로 전달해야 함.
#     # model_common(x, out, model_path)
#
#
# # 입력 2개짜리 1개, 출력 1개
# def simple_model_2(model_path):
#     x = tf.placeholder(tf.int32, shape=[2])
#     out = tf.reduce_sum(x * x)
#
#     model_common([x], [out], model_path)
#
#
# # 입력 1개짜리 2개, 출력 1개
# def simple_model_3(model_path):
#     # 에러. 반드시 shape을 지정해야 함.
#     # x1 = tf.placeholder(tf.int32, shape=[0])
#     # x2 = tf.placeholder(tf.int32, shape=[0])
#
#     x1 = tf.placeholder(tf.int32, shape=[1])
#     x2 = tf.placeholder(tf.int32, shape=[1])
#     out = tf.add(x1, x2)
#
#     # 입력에 2개 전달
#     model_common([x1, x2], [out], model_path)
#
#
# # 입력 1개짜리 2개, 출력 1개짜리 2개
# def simple_model_4(model_path):
#     x1 = tf.placeholder(tf.int32, shape=[1])
#     x2 = tf.placeholder(tf.int32, shape=[1])
#     out_1 = x1 + x2
#     out_2 = x1 * x2
#
#     # 입력에 2개, 출력에 2개 전달
#     model_common([x1, x2], [out_1, out_2], model_path)
#
# model_dir = Path.cwd().parent / "tflite_model"
# simple_model_1('simple_1.tflite')
# simple_model_2('simple_2.tflite')
# simple_model_3('simple_3.tflite')
# simple_model_4('simple_4.tflite')

### tensorflow 2 버전
import tensorflow as tf
from pathlib import Path

def convert_litemodel(func, model_path, input):
    concrete_func = func.get_concrete_function(*input)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(model_path, 'wb') as f:
        f.write(tflite_model)

@tf.function
def simple_model_1(x):
    return x * 5

@tf.function
def simple_model_2(x):
    return tf.reduce_sum(x)

@tf.function
def simple_model_3(x1, x2):
    return x1 + x2

@tf.function
def simple_model_4(x1, x2):
    return x1 + x2, x1 * x2

model_dir = Path.cwd().parent / "tflite_model"
convert_litemodel(simple_model_1, model_dir / 'simple_1.tflite', [tf.TensorSpec(shape=[1], dtype=tf.int32)])
convert_litemodel(simple_model_2, model_dir / 'simple_2.tflite', [tf.TensorSpec(shape=[2], dtype=tf.int32)])
convert_litemodel(simple_model_3, model_dir / 'simple_3.tflite', [tf.TensorSpec(shape=[1], dtype=tf.int32),
                                                                       tf.TensorSpec(shape=[1], dtype=tf.int32)])
convert_litemodel(simple_model_4, model_dir / 'simple_4.tflite', [tf.TensorSpec(shape=[1], dtype=tf.int32),
                                                                       tf.TensorSpec(shape=[1], dtype=tf.int32)])