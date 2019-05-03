package systems.mobile.falldetectionapp;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {
    String modelFile="fallDetection.tflite";
    Interpreter tflite;

    Set<Integer> fallDetectionIds = new HashSet<>(Arrays.asList(0, 1, 2, 4));

    private static final int OUTPUT_SIZE = 6;
    private static final float MIN_PROPABILITY = 0.7f;


    private static final int N_SAMPLES = 100;
    private static List<Float> x;
    private static List<Float> y;
    private static List<Float> z;
    private TextView bscTextView;

    private TextView fklTextView;
    private TextView folTextView;
    private TextView sdlTextView;
    private TextView lyiTextView;
    private TextView stdTextview;
    private TextToSpeech textToSpeech;
    float[][] results = new float[1][OUTPUT_SIZE];
    Vibrator v;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        x = new ArrayList<>();
        y = new ArrayList<>();
        z = new ArrayList<>();

        bscTextView = findViewById(R.id.bsc_prob);
        fklTextView = findViewById(R.id.fkl_prob);
        folTextView = findViewById(R.id.fol_prob);
        sdlTextView = findViewById(R.id.sdl_prob);
        lyiTextView = findViewById(R.id.lyi_prob);
        stdTextview = findViewById(R.id.std_prob);

        try {
            tflite=new Interpreter(loadModelFile(modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);
        v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
    }

    private MappedByteBuffer loadModelFile(String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (results == null || results.length == 0 || results[0].length == 0) {
                    return;
                }

                for (int i = 0; i < results[0].length; i++) {
                    if (Float.compare(results[0][i], MIN_PROPABILITY) > 0 && fallDetectionIds.contains(i)) {
                        textToSpeech.speak("FALL DETECTED", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
                        v.vibrate(500);
                    }
                }
            }
        }, 500, 2500);
    }

    protected void onPause() {
        getSensorManager().unregisterListener(this);
        super.onPause();
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        activityPrediction();
        x.add(event.values[0]);
        y.add(event.values[1]);
        z.add(event.values[2]);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void activityPrediction() {
        if (x.size() == N_SAMPLES && y.size() == N_SAMPLES && z.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(x);
            data.addAll(y);
            data.addAll(z);


            float[] inp = toFloatArray(data);


            tflite.run(inp, results);

            bscTextView.setText(Float.toString(round(results[0][0], 2)));
            fklTextView.setText(Float.toString(round(results[0][1], 2)));
            folTextView.setText(Float.toString(round(results[0][2], 2)));
            sdlTextView.setText(Float.toString(round(results[0][3], 2)));
            lyiTextView.setText(Float.toString(round(results[0][4], 2)));
            stdTextview.setText(Float.toString(round(results[0][5], 2)));

            x.clear();
            y.clear();
            z.clear();
        }
    }

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }
}
