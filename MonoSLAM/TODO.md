The more structured way of doing things would be something like that:

```c++
monoslam::SystemProcessModel processModel = monoslam::SystemProcessModel(linearAccStd, angularAccStd);

monoslam::CameraModel camera = monoslam::CameraModel(K, distCoeffs, frameSize)

monoslam::SystemObservationModel observationModel = monoslam::SystemObservationModel(camera);

monoslam::filter::EKF ekf = monoslam::filter::EKF(processModel, observationModel);

do
{
    frame = device.readFrame();
    const bool ready = monoslam::initSystem(frame, x, P);
    
} while (!ready);

while (true)
{
    ekf.predict(dt);
    ...
    ...
    ...
    frame = device.readframe();
}
```

In update we would then have to do matching to compute the residuals.
