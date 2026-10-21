The more structured way of doing things would be something like that:

```c++
monoslam::SystemProcessModel processModel = monoslam::SystemProcessModel(linearAccStd, angularAccStd);

monoslam::CameraModel camera = monoslam::CameraModel(K, distCoeffs, frameSize)

monoslam::SystemObservationModel observationModel = monoslam::SystemObservationModel(camera);

monoslam::filter::EKF ekf = monoslam::filter::EKF(processModel, observationModel);

do
{
    frame = device.readFrame();
    const bool ready = monoslam::initSystem(frame, x, P, t_init);
    
} while (!ready);

while (true)
{
    dt = time() - t_init

    ekf.predict(dt);
    
    t0 = time()
    frame = device.readframe();
    t1 = time()

    t_init = 0.5 * (t0 + t1); 

    ekf.update([&](const filter::Prediction& p, const cv::Mat1d& S) {
        return matcher.match(frame, p, S);
    });
}
```

In update we would then have to do matching to compute the residuals.
