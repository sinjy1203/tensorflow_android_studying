package org.techtown.android;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;         // 핵심 모듈

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;



public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // xml 파일에 정의된 TextView 객체 얻기
        final TextView tv_output = findViewById(R.id.tv_output);

        // R.id.button_1 : 첫 번째 버튼을 가리키는 id
        // setOnClickListener : 버튼이 눌렸을 때 호출될 함수 설정
        findViewById(R.id.button_1).setOnClickListener(new View.OnClickListener() {
            // 리스너의 기능 중에서 클릭(single touch) 사용
            @Override
            public void onClick(View v) {
                // input : 텐서플로 모델의 placeholder에 전달할 데이터(3)
                // output: 텐서플로 모델로부터 결과를 넘겨받을 배열. 덮어쓰기 때문에 초기값은 의미없다.
                float[][][] input = new float[][][]{{{0}, {0}}, {{0}, {1}}, {{1}, {0}}, {{1}, {1}}};
                float[][] output = new float[][]{{0}, {0}, {0}, {0}};    // 15 = 3 * 5, out = x * 5

                // 1번 모델을 해석할 인터프리터 생성
                Interpreter tflite = getTfliteInterpreter("And_score.tflite");

                // 모델 구동.
                // 정확하게는 from_session 함수의 output_tensors 매개변수에 전달된 연산 호출
                tflite.run(input, output);

                // 출력을 배열에 저장하기 때문에 0번째 요소를 가져와서 문자열로 변환
                tv_output.setText(makeOutputText("hx single", output, null));
            }
        });

        findViewById(R.id.button_2).setOnClickListener(new View.OnClickListener() {
            // 리스너의 기능 중에서 클릭(single touch) 사용
            @Override
            public void onClick(View v) {
                // input : 텐서플로 모델의 placeholder에 전달할 데이터(3)
                // output: 텐서플로 모델로부터 결과를 넘겨받을 배열. 덮어쓰기 때문에 초기값은 의미없다.
                float[][][] input = new float[][][]{{{0}, {0}}, {{0}, {1}}, {{1}, {0}}, {{1}, {1}}};
                float[][] output1 = new float[][]{{0}, {0}, {0}, {0}};    // 15 = 3 * 5, out = x * 5
                int[][] output2 = new int[][]{{0}, {0}, {0}, {0}};

                // 1번 모델을 해석할 인터프리터 생성
                Interpreter tflite = getTfliteInterpreter("And_score.tflite");
                Interpreter tflite2 = getTfliteInterpreter("And_logit.tflite");

                // 모델 구동.
                // 정확하게는 from_session 함수의 output_tensors 매개변수에 전달된 연산 호출
                tflite.run(input, output1);
                tflite2.run(input, output2);

                // 출력을 배열에 저장하기 때문에 0번째 요소를 가져와서 문자열로 변환
                tv_output.setText(makeOutputText("hx single", output1, output2));
            }
        });
    }
    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private String makeOutputText(String title, float[][] output_1, int[][] output_2) {
        String result = title + "\n";
        for (int i = 0; i < output_1.length; i++)
            result += String.valueOf(output_1[i][0]) + " : ";

        if(output_2 != null) {
            result += "\n";
            for (int i = 0; i < output_2.length; i++)
                result += String.valueOf(output_2[i][0]) + " : ";
        }

        return result;
    }
}