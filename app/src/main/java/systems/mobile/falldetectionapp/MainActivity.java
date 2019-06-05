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
import android.util.Log;
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
    String modelFile="fallDetection2.tflite";
    Interpreter tflite;

    Set<Integer> fallDetectionIds = new HashSet<>(Arrays.asList(0, 1, 2, 4));

    private static final int OUTPUT_SIZE = 6;
    private static final float MIN_PROPABILITY = 0.7f;

    private static final int N_SAMPLES = 100;
    private static final int WINDOW_SIZE = 10;
    private static List<Float> a_x;
    private static List<Float> a_y;
    private static List<Float> a_z;
    private static List<Float> g_x;
    private static List<Float> g_y;
    private static List<Float> g_z;
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

        a_x = new ArrayList<>();
        a_y = new ArrayList<>();
        a_z = new ArrayList<>();
        g_x = new ArrayList<>();
        g_y = new ArrayList<>();
        g_z = new ArrayList<>();

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
                        // Tell what fall it was
                        textToSpeech.speak("FALL DETECTED", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
                        v.vibrate(500);
                    }
                }
            }
        }, 500, 2000);
    }

    protected void onPause() {
        getSensorManager().unregisterListener(this);
        super.onPause();
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if(event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            a_x.add(event.values[0]);
            a_y.add(event.values[1]);
            a_z.add(event.values[2]);
            // Log.v("ACC: ",  Float.toString(event.values[0]) + "  " + Float.toString(event.values[1]) + "  " + Float.toString(event.values[2]));

        } else {
            g_x.add(event.values[0]);
            g_y.add(event.values[1]);
            g_z.add(event.values[2]);
            // Log.v("GYRO: ",  Float.toString(event.values[0]) + "  " + Float.toString(event.values[1]) + "  " + Float.toString(event.values[2]));
        }
        if (a_x.size() > N_SAMPLES && a_y.size() > N_SAMPLES && a_z.size() > N_SAMPLES &&  g_x.size() > N_SAMPLES && g_y.size() > N_SAMPLES && g_z.size() > N_SAMPLES) {
            activityPrediction();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {}

    private void activityPrediction() {
        a_x.subList(0, a_x.size() - N_SAMPLES).clear();
        a_y.subList(0, a_y.size() - N_SAMPLES).clear();
        a_z.subList(0, a_z.size() - N_SAMPLES).clear();
        g_x.subList(0, g_x.size() - N_SAMPLES).clear();
        g_y.subList(0, g_y.size() - N_SAMPLES).clear();
        g_z.subList(0, g_z.size() - N_SAMPLES).clear();

        List<Float> data = new ArrayList<>();
        data.addAll(a_x);
        data.addAll(a_y);
        data.addAll(a_z);
        data.addAll(g_x);
        data.addAll(g_y);
        data.addAll(g_z);

        float[] inp = toFloatArray(data);

        tflite.run(inp, results);

        bscTextView.setText(Float.toString(round(results[0][0], 2)));
        fklTextView.setText(Float.toString(round(results[0][1], 2)));
        folTextView.setText(Float.toString(round(results[0][2], 2)));
        sdlTextView.setText(Float.toString(round(results[0][3], 2)));
        lyiTextView.setText(Float.toString(round(results[0][4], 2)));
        stdTextview.setText(Float.toString(round(results[0][5], 2)));

        for (int i=0; i < WINDOW_SIZE; ++i) {
            a_x.remove(0);
            a_y.remove(0);
            a_z.remove(0);
            g_x.remove(0);
            g_y.remove(0);
            g_z.remove(0);
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
