# الخلية 1: التثبيت
# هذا المستوى يتطلب مكتبات معالجة الصور وبيئة Box2D
!apt-get install -y swig
!pip install gymnasium[box2d] stable-baselines3 shimmy

# الخلية 2: الاستيراد
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# الخلية 3: إعداد البيئة
# ملاحظة: continuous=True يعني التحكم المستمر في المقود (أكثر واقعية وصعوبة)
# render_mode="rgb_array" ضروري لعدم تحطم Colab
env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")

# الخلية 4: إعداد النموذج (مساحة عمل الطالب)
# ==============================================================================
# التحدي: المدخلات هنا عبارة عن صور (Pixels) وليست أرقاماً عادية.
# لذلك نستخدم "CnnPolicy" (الشبكات العصبية الالتفافية).
# المطلوب: هذا التدريب بطيء جداً. هل يمكنك ضبط batch_size أو n_steps لتسريعه؟
# ==============================================================================

model = PPO(
    "CnnPolicy",    # ضروري جداً للتعامل مع الصور
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,   # عدد الخطوات التي يجمعها قبل التحديث
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.0    # معامل الانتروبي
)

# الخلية 5: التدريب
# تحذير: بيئة السيارات ثقيلة جداً. التدريب قد يستغرق وقتاً طويلاً (30-60 دقيقة لنتائج أولية).
# نقوم بحفظ نسخة احتياطية كل 5000 خطوة.
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/', name_prefix='racing_model')

print("بدء محرك التدريب... (تحلى بالصبر)")
model.learn(total_timesteps=20000, callback=checkpoint_callback)

# الخلية 6: التقييم والحفظ
from stable_baselines3.common.evaluation import evaluate_policy

# التقييم يأخذ وقتاً أيضاً
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
print(f"متوسط أداء القيادة: {mean_reward}")

model.save("car_racing_final")
print("تم حفظ النموذج النهائي.")

# ملاحظة للطالب: القيادة الناجحة تعني البقاء داخل المسار وعدم الدوران حول النفس.
# النتيجة الإيجابية تعني أن السيارة تتقدم.
