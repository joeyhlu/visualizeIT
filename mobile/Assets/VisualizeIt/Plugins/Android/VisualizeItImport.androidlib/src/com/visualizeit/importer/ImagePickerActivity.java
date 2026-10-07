package com.visualizeit.importer;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import com.unity3d.player.UnityPlayer;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public final class ImagePickerActivity extends Activity {
    private static final int PICK_IMAGE = 41;
    private String callbackTarget;

    public static void pick(Activity activity, String target) {
        activity.runOnUiThread(() -> {
            Intent intent = new Intent(activity, ImagePickerActivity.class);
            intent.putExtra("callbackTarget", target);
            activity.startActivity(intent);
        });
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        callbackTarget = getIntent().getStringExtra("callbackTarget");
        if (state == null) {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.setType("image/*");
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            try { startActivityForResult(intent, PICK_IMAGE); }
            catch (Exception error) { fail("No image picker is available on this phone."); }
        }
    }

    @Override protected void onActivityResult(int request, int result, Intent intent) {
        super.onActivityResult(request, result, intent);
        if (request != PICK_IMAGE) return;
        if (result != RESULT_OK || intent == null || intent.getData() == null) {
            UnityPlayer.UnitySendMessage(callbackTarget, "OnImagePicked", "");
            finish(); return;
        }
        // File IO and decoding must not block the UI thread.
        new Thread(() -> {
            Bitmap bitmap = null;
            try {
                BitmapFactory.Options bounds = new BitmapFactory.Options();
                bounds.inJustDecodeBounds = true;
                try (InputStream stream = getContentResolver().openInputStream(intent.getData())) {
                    BitmapFactory.decodeStream(stream, null, bounds);
                }
                if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw new Exception("Invalid image");
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inSampleSize = 1;
                while (Math.max(bounds.outWidth, bounds.outHeight) / options.inSampleSize > 2048) options.inSampleSize *= 2;
                try (InputStream stream = getContentResolver().openInputStream(intent.getData())) {
                    bitmap = BitmapFactory.decodeStream(stream, null, options);
                }
                if (bitmap == null) throw new Exception("Unable to decode image");
                File file = File.createTempFile("visualizeit-design-", ".png", getCacheDir());
                try (FileOutputStream output = new FileOutputStream(file)) {
                    if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) throw new Exception("Unable to save image");
                }
                String path = file.getAbsolutePath();
                runOnUiThread(() -> { UnityPlayer.UnitySendMessage(callbackTarget,"OnImagePicked",path); finish(); });
            } catch (Exception error) {
                runOnUiThread(() -> fail("Could not open this image. Choose a PNG or JPEG."));
            } finally { if (bitmap != null) bitmap.recycle(); }
        }, "VisualizeIt image import").start();
    }

    private void fail(String message) {
        UnityPlayer.UnitySendMessage(callbackTarget,"OnImagePickFailed",message);
        finish();
    }
}
