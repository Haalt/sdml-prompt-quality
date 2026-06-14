import json


class Tokenizer:
    def init(self, split):
        self.split = split
        # self.map = {}
        self.word_index = {}
        self.length = 1
        self.inverted_map = {}

    def __init__(self, **kwargs):
        self.split = " "
        # self.map = {}
        self.word_index = {}
        self.length = 1
        self.lora_index = {}
        self.lora_length = 1
        self.lora_map = {}
        self.sampler_index = {}
        self.sampler_length = 1
        # self.sampler_map = {}
        self.upscaler_index = {}
        self.upscaler_length = 1
        # self.upscaler_map = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.unknown = set()

        self.generate_inverted_map()
        self.generate_lora_map()
        # self.generate_sampler_map()
        # self.generate_upscaler_map()

    def _get_token(self, string):
        if string.startswith("lora"):
            return string.split(":")[0]
        return string

    def is_lora(self, token):
        return self.inverted_map[token].startswith("lora")

    def fit_on_texts(self, text_arr):
        for text in text_arr:
            if isinstance(text, dict) and "sequence" in text:
                sequence_text = text["sequence"]
            else:
                sequence_text = str(text)
            for _token in sequence_text.split(self.split):
                token = self._get_token(_token)
                if token not in self.word_index:
                    # if token not in self.map:
                    # self.map[token] = True
                    self.word_index[token] = self.length
                    self.length += 1
        self.generate_inverted_map()

    def fit_on_loras(self, text_arr):
        for text in text_arr:
            if isinstance(text, dict) and "sequence" in text:
                # Handle dict format from load_dataset_logits
                sequence_text = text["sequence"]
            else:
                # Handle string format
                sequence_text = str(text)

            for _token in sequence_text.split(self.split):
                token = self._get_token(_token)
                if _token.startswith("lora") and token not in self.lora_index:
                    self.lora_index[token] = self.lora_length
                    self.lora_length += 1
        self.generate_lora_map()

    def generate_inverted_map(self):
        self.inverted_map = {v: k for k, v in self.word_index.items()}
        self.inverted_map[0] = ""  # TODO: check

    def generate_lora_map(self):
        self.lora_map = {v: k for k, v in self.lora_index.items()}
        self.lora_map[0] = ""

    # def generate_sampler_map(self):
    #     self.sampler_map = {v: k for k, v in self.sampler_index.items()}
    # self.sampler_map[0] = ""

    # def generate_upscaler_map(self):
    #     self.upscaler_map = {v: k for k, v in self.upscaler_index.items()}
    #     self.upscaler_map[0] = ""

    def lora_to_token(self, lora_text):
        """Convert LoRA text to token index in the LoRA vocabulary"""
        if lora_text in self.lora_index:
            return self.lora_index[lora_text]
        return 0  # return 0 for unknown LoRAs (padding index)

    def sampler_to_token(self, sampler_text):
        """Convert sampler text to token index in the sampler vocabulary"""
        if sampler_text in self.sampler_index:
            return self.sampler_index[sampler_text]
        print(f"unknown sampler: {sampler_text}")
        return 0  # return 0 for unknown samplers (padding index)

    def upscaler_to_token(self, upscaler_text):
        """Convert upscaler text to token index in the upscaler vocabulary"""
        if upscaler_text in self.upscaler_index:
            return self.upscaler_index[upscaler_text]
        print(f"unknown upscaler: {upscaler_text}")
        return 0  # return 0 for unknown upscalers (padding index)

    def insert_token(self, token, value):
        if token not in self.word_index:
            # if token not in self.map:
            # self.map[token] = True
            self.word_index[token] = value
            self.length = value + 1000  # TODO: remove
        self.generate_inverted_map()

    def token_to_text(
        self,
        token,
    ):
        return self.inverted_map[token]

    def sequence_to_text(self, sequence):
        return [self.token_to_text(token) for token in sequence]

    def texts_to_sequences(self, text_arr):
        seqs = []
        for text in text_arr:
            seq = []
            try:
                # Handle dict format from load_dataset_logits
                # if isinstance(text, dict) and 'sequence' in text:
                text_to_process = text["sequence"]
                # else:
                #     text_to_process = text

                for _token in text_to_process.split(self.split):
                    token = self._get_token(_token)
                    weight = (
                        float(_token.split(":")[1])
                        if _token.startswith("lora")
                        else 1.0
                    )
                    # throws KeyError
                    try:
                        seq.append((self.word_index[token], weight))
                    except KeyError:
                        print("Unknown token: ", token)
                        self.unknown.add(token)
                        raise KeyError
            except KeyError:
                continue
            tmp = {**text}
            tmp['sequence'] = seq
            seqs.append(tmp)
            # seqs.append({"sequence": seq, "cfg_scale": text["cfg_scale"]})
        return seqs

    def __str__(self):
        return json.dumps(self.__dict__)

    def save(self, filename):
        sorted_dict = dict(
            sorted(self.word_index.items(), key=lambda item: item[1]))
        sorted_lora_dict = dict(
            sorted(self.lora_index.items(), key=lambda item: item[1])
        )
        sorted_sampler_dict = dict(
            sorted(self.sampler_index.items(), key=lambda item: item[1])
        )
        sorted_upscaler_dict = dict(
            sorted(self.upscaler_index.items(), key=lambda item: item[1])
        )

        with open(filename, "w") as json_file:
            json.dump(
                {
                    "length": self.length,
                    "split": self.split,
                    "word_index": sorted_dict,
                    "lora_index": sorted_lora_dict,
                    "lora_length": self.lora_length,
                    "sampler_index": sorted_sampler_dict,
                    "sampler_length": self.sampler_length,
                    "upscaler_index": sorted_upscaler_dict,
                    "upscaler_length": self.upscaler_length,
                },
                json_file,
                indent=2,
            )

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "r") as json_file:
            obj_dict = json.load(json_file)

        return cls(**obj_dict)
