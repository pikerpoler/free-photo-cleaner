#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(StorageInfoModule, NSObject)

RCT_EXTERN_METHOD(getStorageInfo:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(batchDelete:(NSArray *)uris resolve:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)

@end
