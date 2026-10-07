#import <UIKit/UIKit.h>
#import <PhotosUI/PhotosUI.h>
#include "UnityInterface.h"

@interface VIImagePicker : NSObject<PHPickerViewControllerDelegate>
@property(nonatomic,copy) NSString *callbackTarget;
@end

static VIImagePicker *activePicker;

@implementation VIImagePicker
- (void)picker:(PHPickerViewController *)picker didFinishPicking:(NSArray<PHPickerResult *> *)results {
    [picker dismissViewControllerAnimated:YES completion:nil];
    NSString *target = self.callbackTarget;
    if (results.count == 0) {
        UnitySendMessage(target.UTF8String,"OnImagePicked","");
        activePicker = nil; return;
    }
    NSItemProvider *provider = results.firstObject.itemProvider;
    if (![provider canLoadObjectOfClass:UIImage.class]) {
        UnitySendMessage(target.UTF8String,"OnImagePickFailed","Choose a PNG or JPEG image.");
        activePicker = nil; return;
    }
    [provider loadObjectOfClass:UIImage.class completionHandler:^(id<NSItemProviderReading> object,NSError *error) {
        UIImage *image = (UIImage *)object;
        NSString *path = nil;
        if (!error && image && image.size.width > 0 && image.size.height > 0) {
            CGFloat scale = MIN(1.0,2048.0/MAX(image.size.width,image.size.height));
            CGSize size = CGSizeMake(image.size.width*scale,image.size.height*scale);
            UIGraphicsBeginImageContextWithOptions(size,NO,1);
            [image drawInRect:CGRectMake(0,0,size.width,size.height)];
            UIImage *normalized = UIGraphicsGetImageFromCurrentImageContext();
            UIGraphicsEndImageContext();
            NSData *data = UIImagePNGRepresentation(normalized);
            NSString *candidate = [NSTemporaryDirectory() stringByAppendingPathComponent:
                [NSString stringWithFormat:@"visualizeit-design-%@.png",NSUUID.UUID.UUIDString]];
            if (data && [data writeToFile:candidate atomically:YES]) path = candidate;
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            if (path) UnitySendMessage(target.UTF8String,"OnImagePicked",path.UTF8String);
            else UnitySendMessage(target.UTF8String,"OnImagePickFailed","Could not open this image. Try another design.");
            activePicker = nil;
        });
    }];
}
@end

extern "C" void VI_PickImage(const char *target) {
    NSString *callback = [NSString stringWithUTF8String:target];
    dispatch_async(dispatch_get_main_queue(), ^{
        if (activePicker) return;
        activePicker = [VIImagePicker new];
        activePicker.callbackTarget = callback;
        PHPickerConfiguration *configuration = [[PHPickerConfiguration alloc] init];
        configuration.filter = PHPickerFilter.imagesFilter;
        configuration.selectionLimit = 1;
        PHPickerViewController *picker = [[PHPickerViewController alloc] initWithConfiguration:configuration];
        picker.delegate = activePicker;
        [UnityGetGLViewController() presentViewController:picker animated:YES completion:nil];
    });
}
