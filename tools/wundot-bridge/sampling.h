// sampling.h

#ifndef SAMPLING_H
#define SAMPLING_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sampling_params {
    float temp;
    float top_p;
    int   top_k;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;
    int   mirostat;
    int   n_predict;
} sampling_params;

#ifdef __cplusplus
}
#endif

#endif  // SAMPLING_H
