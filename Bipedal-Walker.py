# الخلية 1: التثبيت
!apt-get install -y swig
!pip install gymnasium[box2d] stable-baselines3

# الخلية 2: الاستيراد
import gymnasium as gym
from stable_baselines3 import SAC  # Soft Actor-Critic

# الخلية 3: إعداد البيئة
# ملاحظة: hardcore=False للنسخة العادية. النسخة الـ Hardcore تحتوي على حفر وسلالم.
env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")

# الخلية 4: الخوارزمية (مساحة عمل الطالب)
# ==============================================================================
# التحدي: الفضاء الحركي متصل (Continuous Action Space) ومعقد.
# نستخدم خوارزمية SAC لأنها فعالة جداً في هذه الحالات.
# المطلوب: هل يمكنك تعديل ent_coef لجعل الروبوت يستكشف حركات غريبة قبل الاستقرار؟
# ==============================================================================

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=300000,
    batch_size=256,
    ent_coef='auto',   # تلقائي لضبط التوازن بين الاستكشاف والاستغلال
    train_freq=1,
    gradient_steps=1,
    learning_starts=10000
)

# الخلية 5: التدريب الطويل
print("جاري تعليم الروبوت المشي... (سيستغرق وقتاً)")
model.learn(total_timesteps=60000)

# الخلية 6: التقييم
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"متوسط المكافأة: {mean_reward}")

# الهدف: الوصول لـ 300 نقطة يعني أن الروبوت يجري!
if mean_reward > 250:
    print("النتيجة: الروبوت يمشي بثبات!")
else:
    print("النتيجة: الروبوت يتعثر، يحتاج لمزيد من التدريب.")
