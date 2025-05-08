Update session_create in C++
Modify the session to use the already-loaded model and ctx if they exist globally. I’ll assume you want to:

Call LoadModel() once to initialize the backend and model.

Reuse that model in each NewSession() call.



Modify bridge.go for shared model context
We’ll add:

A shared model pointer sharedModelPtr and context sharedCtxPtr

Updated NewSession to clone from shared context/model
