# الخلية 1: التثبيت
!apt-get install -y swig
!pip install gymnasium[box2d] stable-baselines3

# الخلية 2: الاستيراد
import gymnasium as gym
from stable_baselines3 import DQN

# الخلية 3: ضبط المعاملات العليا (مساحة عمل الطالب)
# ==============================================================================
# التحدي: المركبة تتحطم أو تستهلك وقوداً زائداً.
# المطلوب: تغيير القيم في القاموس أدناه لتحسين أداء الوكيل.
# تلميح: جرب تغيير learning_rate و batch_size.
# ==============================================================================

# إعدادات النموذج
hyperparameters = {
    "learning_rate": 1e-3,          # جرب قيم مثل: 5e-4, 1e-4
    "batch_size": 64,               # جرب: 128
    "buffer_size": 50000,           # حجم ذاكرة الخبرات
    "learning_starts": 1000,        # متى يبدأ التعلم
    "gamma": 0.99,                  # أهمية المكافآت المستقبلية
    "target_update_interval": 250,  # معدل تحديث الشبكة المستهدفة
    "train_freq": 4,
    "exploration_fraction": 0.15,   # نسبة الاستكشاف
    "exploration_final_eps": 0.05
}

# الخلية 4: تهيئة البيئة والنموذج
env = gym.make("LunarLander-v2", render_mode="rgb_array")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    **hyperparameters
)

# الخلية 5: التدريب
# يحتاج هذا المستوى وقتاً أطول للتدريب
print("بداية التدريب (قد يستغرق بضع دقائق)...")
model.learn(total_timesteps=80000)

# الخلية 6: التقييم والحفظ
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

print(f"متوسط نقاط الهبوط: {mean_reward}")
# الهدف: الحصول على 200 نقطة أو أكثر يعتبر حلاً مثالياً.
if mean_reward >= 200:
    print("النتيجة: هبوط ناجح ومثالي!")
elif mean_reward >= 0:
    print("النتيجة: هبوط آمن ولكن يحتاج تحسين.")
else:
    print("النتيجة: تحطم المركبة.")
