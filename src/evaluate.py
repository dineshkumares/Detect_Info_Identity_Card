import codecs
import editdistance

PREDICT = "../data/textPredict.txt"
TRUTH = "../data/textTruth.txt"


def estDistance(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    return editdistance.eval(s1, s2) / max(len(s1), len(s2))


def evalute(predict, truth):
    fPredict = codecs.open(predict, "r", "utf8")
    fTruth = codecs.open(truth, "r", "utf8")
    lPre = fPredict.read().split("\n")
    lTru = fTruth.read().split("\n")

    totalDis = 0

    if len(lPre) == len(lTru):
        numElement = len(lPre)
        for i in range(len(lPre)):
            totalDis = totalDis + (1.0 - estDistance(lPre[i], lTru[i]))
    else:
        numElement = min(len(lPre), len(lTru))
        for i in range(numElement):
            totalDis = totalDis + (1 - estDistance(lPre[i], lTru[i]))

    return totalDis / numElement


if __name__ == '__main__':
    print(evalute(PREDICT, TRUTH))
