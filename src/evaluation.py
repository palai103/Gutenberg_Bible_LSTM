import pathlib
from datetime import datetime
import json
import os
import string
import unicodedata
import editdistance

import Levenshtein as lev
import cv2
import matplotlib.pyplot as plt


latin_words = []


def load_latin_words(datasets):
    for dataset in datasets:
        f = open("../dataset/" + dataset + ".txt", "r")

        for line in f:
            for word in line.split(" "):
                word = word.replace("\n", "")
                if word[0] != "_" and word not in latin_words:
                    latin_words.append(word)

        f.close()


def save_json(filename, data):
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)


def decode_predicted_output(outputs, labels, images, ground_truth, dictionary, fold_index=""):
    image_number = 0

    raw_predicts = []
    predicts = []
    ground_truths = []

    results = pathlib.Path("./results")
    results.mkdir(parents=True, exist_ok=True)

    timestamp = str(datetime.now()).replace(":", "").split(".")[0]

    images_folder = "./results/images" + str(fold_index) + "-" + timestamp

    os.mkdir(images_folder)

    for output in outputs:
        line_index = outputs.index(output)
        text = ""
        correct_text = ""

        output_aslist = output.tolist()

        for number in output_aslist[0]:
            text += dictionary[str(number)]

        predicted = text
        raw_predicts.append(predicted)
        text = adjust_prediction(text)

        for number in labels[line_index]:
            correct_text += dictionary[str(int(number))]

        correct_text = correct_text.replace("-", "")
        image_index = ground_truth.tolist().index(labels[line_index].tolist())
        image = images[image_index]

        fig = plt.figure(figsize=(1.5, 0.5), dpi=300)
        newax = fig.add_axes([0.15, 0, 0.7, 0.7], anchor='N')
        newax.imshow(cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2RGB))
        newax.axis('off')
        plt.axis('off')
        plt.gcf().text(0.01, 0.05, correct_text, fontsize=4, color='green')
        plt.gcf().text(0.01, 0.2, text, fontsize=4, color='red')
        plt.gcf().text(0.01, 0.35, predicted, fontsize=4, color='blue')
        plt.savefig(images_folder + '/prediction_' + str(image_number) + '.png', bbox_inches='tight')
        plt.close()
        image_number += 1

        print("Prediction: " + text)
        print("Correct label: " + correct_text)

        predicts.append(text)
        ground_truths.append(correct_text)

        print("-----")

    raw_metrics = ocr_metrics(raw_predicts, ground_truths)

    print("Raw CER: " + str(raw_metrics[0]))
    print("Raw WER: " + str(raw_metrics[1]))
    print("Raw SER: " + str(raw_metrics[2]))

    metrics = ocr_metrics(predicts, ground_truths)

    print("CER: " + str(metrics[0]))
    print("WER: " + str(metrics[1]))
    print("SER: " + str(metrics[2]))

    results = {"Raw CER": str(raw_metrics[0]),
               "Raw WER": str(raw_metrics[1]),
               "Raw SER": str(raw_metrics[2]),
               "CER": str(metrics[0]),
               "WER": str(metrics[1]),
               "SER": str(metrics[2])}

    save_json("./results/results" + str(fold_index) + "-" + timestamp + ".json", results)


def adjust_prediction(prediction):
    adjusted_prediction = []

    for word in prediction.split(" "):
        lowest_distance = 999
        current_adjustment = ""

        for dictionary_word in latin_words:
            dictionary_word = dictionary_word.replace("\n", "")
            distance = lev.distance(word, dictionary_word)

            if distance < lowest_distance:
                lowest_distance = distance
                current_adjustment = dictionary_word

        adjusted_prediction.append(current_adjustment)

    return " ".join(adjusted_prediction)


def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return 1, 1, 1

    wrong_chars = 0
    total_chars = 0

    wrong_words = 0
    total_words = 0

    wrong_sentences = 0
    total_sentences = len(predicts)

    for (pd, gt) in zip(predicts, ground_truth):
        pd, gt = pd.lower(), gt.lower()

        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        wrong_chars += editdistance.eval(pd_cer, gt_cer)
        total_chars += max(len(pd_cer), len(gt_cer))

        pd_wer, gt_wer = pd.split(), gt.split()
        wrong_words += editdistance.eval(pd_wer, gt_wer)
        total_words += max(len(pd_wer), len(gt_wer))

        pd_ser, gt_ser = [pd], [gt]
        wrong_sentences += editdistance.eval(pd_ser, gt_ser)

    metrics = [wrong_chars/total_chars, wrong_words/total_words, wrong_sentences/total_sentences]

    return metrics
