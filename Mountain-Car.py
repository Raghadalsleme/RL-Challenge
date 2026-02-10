# الخلية 1: التثبيت
!apt-get install -y swig
!pip install gymnasium[box2d] stable-baselines3 shimmy

# الخلية 2: الاستيراد
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# الخلية 3: هندسة دالة المكافأة (مساحة عمل الطالب)
# ==============================================================================
# التحدي: المكافأة الافتراضية هي -1 لكل خطوة، مما يجعل التعلم بطيئاً جداً.
# المطلوب: تعديل دالة `step` أدناه لإضافة "مكافأة تشجيعية" تعتمد على سرعة أو موقع العربة.
# ==============================================================================

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # استخراج خصائص الحالة
        position = next_state[0] # موقع العربة
        velocity = next_state[1] # سرعة العربة

        # --- بداية كود الطالب ---
        # قم بكتابة منطق المكافأة الجديد هنا.
        # مثال مقترح (قم بتفعيله أو كتابة غيره):
        # reward += abs(velocity) * 10  # مكافأة تزيد كلما زادت السرعة
        
        # --- نهاية كود الطالب ---
        
        return next_state, reward, terminated, truncated, info

# الخلية 4: إعداد البيئة والنموذج
base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = CustomRewardWrapper(base_env)

# استخدام خوارزمية DQN
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.001,
    buffer_size=10000,
    exploration_fraction=0.2
)

# الخلية 5: التدريب
print("جاري تدريب النموذج...")
model.learn(total_timesteps=30000)

# الخلية 6: التقييم
# الهدف: الحصول على متوسط مكافأة أعلى من -110
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"متوسط المكافأة: {mean_reward} +/- {std_reward}")

if mean_reward > -110:
    print("الحالة: تم اجتياز المستوى الأول بنجاح.")
else:
    print("الحالة: لم يتم الاجتياز، حاول تحسين دالة المكافأة.")
