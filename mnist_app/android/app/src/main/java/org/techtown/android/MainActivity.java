package org.techtown.android;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {
    private static final int FROM_ALBUM = 1;    // onActivityResult 식별자
    private static final int FROM_CAMERA = 2;   // 카메라는 사용 안함

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 인텐트의 결과는 onActivityResult 함수에서 수신.
        // 여러 개의 인텐트를 동시에 사용하기 때문에 숫자를 통해 결과 식별(FROM_ALBUM 등등)
        findViewById(R.id.button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setType("image/*");                      // 이미지만
                intent.setAction(Intent.ACTION_GET_CONTENT);    // 카메라(ACTION_IMAGE_CAPTURE)
                startActivityForResult(intent, FROM_ALBUM);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // 카메라를 다루지 않기 때문에 앨범 상수에 대해서 성공한 경우에 대해서만 처리
        if (requestCode != FROM_ALBUM || resultCode != RESULT_OK)
            return;

        try {
            // 선택한 이미지에서 비트맵 생성
            InputStream stream = getContentResolver().openInputStream(data.getData());
            Bitmap bmp = BitmapFactory.decodeStream(stream);
            stream.close();

            ImageView iv = findViewById(R.id.photo);
            iv.setScaleType(ImageView.ScaleType.FIT_XY);    // [300, 300]에 꽉 차게 표시
            iv.setImageBitmap(bmp);

            // ---------------------------------------- //
            // 검증 코드. 여러 차례 변환하기 때문에 PC 버전과 같은지 확인하기 위해 사용.

            // mnist 원본은 0~1 사이의 실수를 사용해 픽셀을 표현한다. 픽셀 1개에 1바이트가 아니라 4바이트 사용.
            // 메모리 용량은 3136(28 * 28 * 4). 입력 이미지를 똑같이 만들어서 전달해야 한다.

            // mnist에서 생성한 숫자 이미지 파일이 흑백이긴 하지만 ARGB를 사용해서 색상을 표시하기 때문에
            // 가운데 픽셀의 경우 fffcfcfc와 같은 형태로 나온다.
            // ff는 alpha를 가리키고 동일한 값인 fc가 RGB에 공통으로 나온다.

            // getPixel 함수는 int를 반환하기 때문에 부호 없는 16진수로 확인해야 한다.
            // int pixel = bmp.getPixel(14, 14);
            // Log.d("getPixel", Integer.toUnsignedString(pixel, 16));

            // 원본 mnist 이미지는 (28, 28, 1)로 되어 있다.
            // getByteCount 함수로 확인해 보면 3136으로 나오는데
            // 각각의 픽셀이 4바이트로 구성되어 있기 때문에 그렇다. 784 * 4 = 3136
            // int bytes = bmp.getByteCount();
            // Log.d("getByteCount", Integer.toString(bytes));

            // mnist 원본 이미지와 비교하기 위해 줄 단위로 변환 결과 출력
            // 파이썬에서 똑같은 파일을 읽어들여서 에뮬레이터 출력과 비교. 똑같이 나온다. 성공.
            // 2차원 배열을 한 번에 깔끔하게 출력할 수 없기 때문에 아래 코드가 필요하다.
            // float[] row = new float[28];
            // for(int y = 0; y < 28; y++) {
            //     for(int x = 0; x < 28; x++) {
            //         int pixel = bmp.getPixel(x, y);         // x가 앞쪽, y가 뒤쪽.
            //         row[x] = (pixel & 0xff) / (float) 255;  // 실수 변환하지 않으면 0과 1로만 나온다.
            //     }
            //     // 줄 단위 출력. 그래도 자릿수가 맞지 않아 numpy처럼 나오진 않는다.
            //     Log.d(String.format("%02d", y), Arrays.toString(row));
            // }

            // ---------------------------------------- //

            // 비트맵 이미지로부터 RGB에 해당하는 값을 1개만 가져와서
            // mnist 원본과 동일하게 0~1 사이의 실수로 변환하고, 1차원 784로 만들어야 한다.
            // 그러나, 실제로 예측할 때는 여러 장을 한 번에 전달할 수 있어야 하기 때문에
            // 아래와 같이 2차원 배열로 만드는 것이 맞다.
            // 만약 1장에 대해서만 예측을 하고 싶다면 1차원 배열로 만들어도 동작한다.
            float[][] bytes_img = new float[1][784];

            for(int y = 0; y < 28; y++) {
                for(int x = 0; x < 28; x++) {
                    int pixel = bmp.getPixel(x, y);
                    bytes_img[0][y*28+x] = (pixel & 0xff) / (float) 255;
                }
            }

            // 파이썬에서 만든 모델 파일 로딩
            Interpreter tf_lite = getTfliteInterpreter("mnist.tflite");

            // 케라스로부터 변환할 때는 입력이 명시되지 않기 때문에 입력을 명확하게 정의할 필요가 있다.
            // 이때 getInputTensor 함수를 사용한다. getOutputTensor 함수도 있다.
            // 입력은 1개밖에 제공하지 않았고, 784의 크기를 갖는 1차원 이미지.
            // 입력이 1개라는 뜻은 getInputTensor 함수에 전달할 인덱스가 0밖에 없다는 뜻이다.
            // 여러 장의 이미지를 사용하면 shape에 표시된 1 대신 이미지 개수가 들어간다.
            // input : [1, 784]
            // Tensor input = tf_lite.getInputTensor(0);
            // Log.d("input", Arrays.toString(input.shape()));

            // 출력 배열 생성. 1개만 예측하기 때문에 [1] 사용
            // bytes_img에서처럼 1차원으로 해도 될 것 같은데, 여기서는 에러.
            float[][] output = new float[1][10];
            tf_lite.run(bytes_img, output);

            Log.d("predict", Arrays.toString(output[0]));

            // 텍스트뷰 10개. 0~9 사이의 숫자 예측
            int[] id_array = {R.id.result_0, R.id.result_1, R.id.result_2, R.id.result_3, R.id.result_4,
                    R.id.result_5, R.id.result_6, R.id.result_7, R.id.result_8, R.id.result_9};

            for(int i = 0; i < 10; i++) {
                TextView tv = findViewById(id_array[i]);
                tv.setText(String.format("%.5f", output[0][i]));    // [0] : 2차원 배열의 첫 번째
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 모델 파일 인터프리터를 생성하는 공통 함수
    // loadModelFile 함수에 예외가 포함되어 있기 때문에 반드시 try, catch 블록이 필요하다.
    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    // 모델을 읽어오는 함수로, 텐서플로 라이트 홈페이지에 있다.
    // MappedByteBuffer 바이트 버퍼를 Interpreter 객체에 전달하면 모델 해석을 할 수 있다.
    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}