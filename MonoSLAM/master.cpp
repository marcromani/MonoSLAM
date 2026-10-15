

processModel = monoslam::ProcessModel(linAcc, angAcc);
observationModel = monoslam::ObservationModel(K);

ekf = monoslam::filter::EKF();

do {
    frame = device.readFrame();
    const bool ready = monoslam::initSystem(frame, x, P);
} while (!ready);

while(true) {
    ekf.predict(x, P, processModel);
    ekf.update(x, P, observationModel);

    frame = device.readframe();
    e
}



in update we have to match to compute the residuals
