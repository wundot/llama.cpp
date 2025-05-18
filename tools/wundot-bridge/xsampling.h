// sampling.h

#ifndef XSAMPLING_H
#define XSAMPLING_H

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

#endif  // XSAMPLING_H
