# الخلية 1: إعداد البيئة وتثبيت المكتبات
# يقوم هذا الكود بتثبيت المكتبات الأساسية للعمل.
!pip install gymnasium numpy matplotlib

import gymnasium as gym
import numpy as np
import random

# الخلية 2: تكوين البيئة
# ملاحظة للطلاب: البيئة حالياً غير زلقة (is_slippery=False).
# للتحدي الحقيقي، يجب تغييرها إلى True ومحاولة حلها.
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")

# الخلية 3: إعدادات الخوارزمية (مساحة عمل الطالب)
# ==============================================================================
# المطلوب: قم بضبط المعاملات العليا (Hyperparameters) أدناه للوصول لأفضل نتيجة.
# الهدف: تدريب الوكيل للوصول إلى الهدف دون الوقوع في الحفر.
# ==============================================================================

# المعاملات العليا
total_episodes = 2000         # عدد جولات التدريب
learning_rate = 0.8           # معدل التعلم (Alpha)
max_steps = 99                # أقصى عدد خطوات في الجولة الواحدة
gamma = 0.95                  # معامل الخصم (Discount Factor)

# معاملات الاستكشاف (Exploration Parameters)
epsilon = 1.0                 # احتمالية الاستكشاف الأولية
max_epsilon = 1.0             # الحد الأعلى للاستكشاف
min_epsilon = 0.01            # الحد الأدنى للاستكشاف
decay_rate = 0.005            # معدل اضمحلال الاستكشاف

# تهيئة جدول Q-Table بالاصفار
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

# الخلية 4: حلقة التدريب (Q-Learning Algorithm)
print("بداية التدريب...")
rewards = []

for episode in range(total_episodes):
    state, info = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # 1. اختيار الإجراء بناءً على استراتيجية Epsilon-Greedy
        tradeoff = random.uniform(0, 1)
        
        if tradeoff > epsilon:
            action = np.argmax(qtable[state, :]) # استغلال المعرفة الحالية (Exploitation)
        else:
            action = env.action_space.sample()   # استكشاف عشوائي (Exploration)

        # 2. تنفيذ الإجراء ومراقبة النتيجة
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. تحديث جدول Q باستخدام معادلة بيلمان (Bellman Equation)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        state = new_state
        
        if done:
            break
            
    # تقليل نسبة الاستكشاف تدريجياً
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print(f"تم الانتهاء من التدريب على {total_episodes} جولة.")
print(f"متوسط المكافأة في آخر 100 جولة: {sum(rewards[-100:])/100}")

# الخلية 5: التقييم النهائي
# ملاحظة: النجاح في هذه البيئة يعني الحصول على متوسط مكافأة 1.0 (في حالة عدم الانزلاق).
if sum(rewards[-100:])/100 >= 1.0:
    print("النتيجة: نجاح (تم حل البيئة).")
else:
    print("النتيجة: فشل (حاول تعديل المعاملات العليا).")
