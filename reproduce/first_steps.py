"""在终端跟算讲义第 8、10、16 页。只需 Python 标准库与同目录的实验源码。"""
import argparse
import math

from prediction_lab import exercise, read_prediction


def pixel(noise_guess):
    """用训练程序知道的雪花出题，再用预测的雪花还原一个像素。"""
    clean, noise, signal = 0.8, -1.0, 0.8
    state, _ = exercise(clean, noise, signal)
    clean_guess, _ = read_prediction(state, noise_guess, "epsilon", signal)
    return state, clean_guess


def crossing(upper_probability):
    """两道半程相交的题，分别给出向右上、向右下的标签。"""
    upper, lower = (4, 2), (4, -2)
    p = upper_probability
    mean = tuple(p*a + (1-p)*b for a,b in zip(upper, lower))

    def loss(guess):
        upper_error = sum((g-u)**2 for g,u in zip(guess,upper))
        lower_error = sum((g-u)**2 for g,u in zip(guess,lower))
        return p*upper_error + (1-p)*lower_error

    return mean, loss(upper), loss(mean)


def gate_updates():
    """冻结基座；固定支路输入 r，观察出口更新前后，损失能否传到 r。"""
    base, r, target = 0.2, 2.0, 1.0
    weight, bias, learning_rate = 0.0, 0.0, 0.1
    for step in range(2):
        prediction = base + weight*r + bias
        error = prediction-target                  # 损失为 (预测−目标)²/2
        grad_weight, grad_bias = error*r, error
        grad_r = error*weight
        yield step, weight, bias, prediction, grad_weight, grad_r
        weight -= learning_rate*grad_weight
        bias -= learning_rate*grad_bias


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-guess", type=float, default=-0.7, help="第 8 页预测的噪声")
    parser.add_argument("--upper", type=float, default=.5, help="第 10 页抽中右上箭头的概率")
    args=parser.parse_args()
    if not math.isfinite(args.noise_guess) or not 0<=args.upper<=1:
        parser.error("噪声猜测需要是有限数，概率需要在 0 与 1 之间")
    state, clean_guess = pixel(args.noise_guess)
    print(f"第 8 页　带噪输入 {state:.3f}，噪声猜测 {args.noise_guess:.3f}，干净估计 {clean_guess:.3f}")
    mean, first_loss, best_loss = crossing(args.upper)
    print(f"第 10 页　平均箭头 ({mean[0]:.3f}, {mean[1]:.3f})，猜右上损失 {first_loss:.3f}，猜平均损失 {best_loss:.3f}")
    for step, weight, bias, pred, grad_weight, grad_r in gate_updates():
        print(f"第 16 页　已更新 {step} 次，W={weight:.3f}，b={bias:.3f}，预测 {pred:.3f}，"
              f"出口梯度 dL/dW={grad_weight:.3f}，支路输入梯度 dL/dr={grad_r+0.0:.3f}")


if __name__=="__main__":
    main()
