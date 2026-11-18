# AcceleratedKAZE

A managed .NET implementation of the AKAZE feature detector and the TEBLID descriptor. Keypoints and descriptors match OpenCV 4.12's implementation.

```csharp
using var session = new Akaze.Session();
session.Compute(gray, width, height);
new Teblid(Teblid.AkazeScale).Compute(session, gray, width, height, descriptors);
```
The code is MIT licensed. It is derived from [OpenCV](https://github.com/opencv/opencv); the TEBLID weights come from OpenCV contrib under the [Apache License 2.0](LICENSE-OPENCV).
