# الخلية 1: التثبيت
!pip install gymnasium stable-baselines3

# الخلية 2: الاستيراد
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# الخلية 3: صناعة البيئة المستحيلة (مساحة عمل الطالب)
# ==============================================================================
# التحدي: في هذه البيئة، الجاذبية مضاعفة والعصا قصيرة جداً مما يجعلها تسقط بسرعة هائلة.
# المطلوب: يجب عليك ضبط الشبكة العصبية لتكون ردة فعلها سريعة جداً.
# ==============================================================================

# إنشاء البيئة الأساسية
env = gym.make("CartPole-v1", render_mode="rgb_array")

# تعديل الفيزياء لجعلها "مستحيلة"
env.unwrapped.gravity = 25.0       # جاذبية قوية جداً (الطبيعي 9.8)
env.unwrapped.length = 0.05        # عصا قصيرة جداً (سريعة السقوط)
env.unwrapped.force_mag = 30.0     # قوة دفع محرك العربة

# الخلية 4: النموذج
# نستخدم PPO لقدرته على التكيف، لكن يجب ضبط معاملاته بدقة.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,  # معدل تعلم يحتاج لتجربة
    n_steps=1024,
    batch_size=64,
    gamma=0.98,
    gae_lambda=0.95
)

# الخلية 5: التدريب تحت الضغط
print("بدء تحدي التوازن الفائق...")
model.learn(total_timesteps=100000)

# الخلية 6: اختبار النتيجة النهائية
# النجاح في هذا المستوى يعتبر إنجازاً كبيراً.
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"النتيجة النهائية: {mean_reward}")

if mean_reward > 400:
    print("التقييم: عبقري! (فائق الصعوبة مكتمل)")
elif mean_reward > 200:
    print("التقييم: جيد جداً، توازن مقبول.")
else:
    print("التقييم: فشل التوازن، الفيزياء قوية جداً.")
