package com.example.skincancer;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private TextView resultTextView;
    private Module model;
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<Intent> galleryLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);

        // Charger le modèle PyTorch
        try {
            model = Module.load(assetFilePath("model_scripted.pt"));
        } catch (IOException e) {
            resultTextView.setText("Erreur : impossible de charger le modèle.");
            e.printStackTrace();
        }

        // Initialisation des launchers pour la caméra et la galerie
        initLaunchers();

        // Capture d'image via la caméra
        Button captureButton = findViewById(R.id.captureButton);
        captureButton.setOnClickListener(v -> openCamera());

        // Choisir une image depuis la galerie
        Button galleryButton = findViewById(R.id.galleryButton);
        galleryButton.setOnClickListener(v -> openGallery());
    }

    private void initLaunchers() {
        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Bitmap bitmap = (Bitmap) result.getData().getExtras().get("data");
                        handleImageProcessing(bitmap, "Erreur : Impossible de récupérer l'image de la caméra.");
                    } else {
                        resultTextView.setText("Capture annulée.");
                    }
                });

        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri selectedImage = result.getData().getData();
                        try {
                            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                            handleImageProcessing(bitmap, "Erreur : Impossible de récupérer l'image de la galerie.");
                        } catch (IOException e) {
                            resultTextView.setText("Erreur : " + e.getMessage());
                            e.printStackTrace();
                        }
                    } else {
                        resultTextView.setText("Sélection annulée.");
                    }
                });
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(intent);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryLauncher.launch(intent);
    }

    private void handleImageProcessing(Bitmap bitmap, String errorMessage) {
        if (bitmap == null) {
            resultTextView.setText(errorMessage);
            return;
        }

        // Afficher l'image
        imageView.setImageBitmap(bitmap);

        // Prétraitement et inférence
        try {
            bitmap = resizeBitmap(bitmap, 224, 224);
            Tensor inputTensor = preprocessImage(bitmap);
            Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();

            // Extraire les scores de prédiction (1000 éléments pour ImageNet)
            float[] resultArray = outputTensor.getDataAsFloatArray();

            // Debug: Afficher les valeurs des résultats avant le calcul de la probabilité
            StringBuilder debugOutput = new StringBuilder();
            for (float value : resultArray) {
                debugOutput.append(value).append(" ");
            }
            resultTextView.setText("Model Output: " + debugOutput.toString());

            // Extraire le premier score pour la classification binaire
            float predictionScore = resultArray[0];  // Utilisation du premier score pour la classification

            // Appliquer la fonction sigmoïde pour obtenir une probabilité
            float probability = (float) (1 / (1 + Math.exp(-predictionScore)));  // Sigmoïde

            // Logique des seuils pour prédiction (modification du seuil à 0.3)
            String prediction;
            if (probability > 0.3) {  // Utilisation d'un seuil de 0.3
                prediction = "Cancer malin détecté avec une confiance de : " + (probability * 100) + "%";
            } else {
                prediction = "Pas de cancer détecté avec une confiance de : " + ((1 - probability) * 100) + "%";
            }

            // Afficher le résultat
            resultTextView.setText(prediction);

        } catch (Exception e) {
            resultTextView.setText("Erreur lors de l'inférence du modèle : " + e.getMessage());
            e.printStackTrace();
        }
    }

    private Tensor preprocessImage(Bitmap bitmap) {
        return TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                new float[]{0.485f, 0.456f, 0.406f}, // Moyenne pour la normalisation
                new float[]{0.229f, 0.224f, 0.225f}  // Écart-type pour la normalisation
        );
    }

    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (!file.exists()) {
            try (InputStream is = getAssets().open(assetName);
                 FileOutputStream fos = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int length;
                while ((length = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, length);
                }
            }
        }
        return file.getAbsolutePath();
    }

    private Bitmap resizeBitmap(Bitmap bitmap, int width, int height) {
        return Bitmap.createScaledBitmap(bitmap, width, height, true);
    }
}
