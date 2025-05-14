# coding:utf-8


def vad(audio_file):
    pass


def demo():
    from funasr import AutoModel

    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      spk_model="cam++", spk_model_revision="v2.0.2",
                      disable_update=True,
                      )
    pass

