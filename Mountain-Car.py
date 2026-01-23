# ========================================================================
# تحدي التعلم الذاتي (للمتخصصين)
# المستوى الأول: تسلق الجبل (Mountain Car)
# الصعوبة: سهل (مبتدئ)
# ========================================================================

"""
 وصف التحدي:
--------------
سيارة عالقة في وادٍ بين جبلين. الهدف هو الوصول إلى العلم على قمة الجبل الأيمن.
المشكلة: محرك السيارة ضعيف ولا يستطيع الصعود مباشرة!
الحل: يجب على السيارة التأرجح ذهاباً وإياباً لبناء الزخم والوصول للقمة.

 القوانين والقيود:
-------------------
1. يجب استخدام خوارزمية Q-Learning فقط
2. لا يسمح باستخدام Deep Learning أو Neural Networks
3. يجب تقسيم الحالات (State Discretization) لأن الفضاء مستمر
4. الإجراءات المسموحة: يسار (0)، لا شيء (1)، يمين (2)
5. النجاح = الوصول للعلم في أقل من 200 خطوة

🏆 معايير التقييم:
------------------
- نقاط إضافية: الوصول للهدف في أقل عدد خطوات
- يتم احتساب المجموع النهائي بناءً على متوسط آخر 100 حلقة
- الفائز: الفريق الذي يحصل على أعلى مجموع كلي للنقاط

⚠️ تنبيهات هامة:
-----------------
- لا تقم بتعديل البيئة (Environment) أو قوانين المكافآت
- يمكنك فقط تعديل معاملات التعلم وطريقة تقسيم الحالات
- يجب عليك تقديم الكود المصدري مع شرح الاستراتيجية
"""

# ========================================================================
# 1️⃣ استيراد المكتبات المطلوبة
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import gymnasium as gym
from collections import defaultdict

# ========================================================================
# 2️⃣ إعداد البيئة (لا تعدل هذا القسم!)
# ========================================================================

class MountainCarChallenge:
    """
    بيئة تحدي Mountain Car
    ⚠️ ممنوع التعديل على هذا الكلاس!
    """
    
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.position_bins = None
        self.velocity_bins = None
        
    def setup_discretization(self, n_position_bins=20, n_velocity_bins=20):
        """
        إعداد تقسيم الحالات المستمرة إلى حالات منفصلة
        
        المعاملات:
        -----------
        n_position_bins: عدد الأقسام للموقع
        n_velocity_bins: عدد الأقسام للسرعة
        """
        position_space = np.linspace(-1.2, 0.6, n_position_bins)
        velocity_space = np.linspace(-0.07, 0.07, n_velocity_bins)
        
        self.position_bins = position_space
        self.velocity_bins = velocity_space
    
    def discretize_state(self, state):
        """تحويل الحالة المستمرة إلى منفصلة"""
        position, velocity = state
        
        position_idx = np.digitize(position, self.position_bins)
        velocity_idx = np.digitize(velocity, self.velocity_bins)
        
        return (position_idx, velocity_idx)
    
    def reset(self):
        """إعادة تعيين البيئة"""
        state, _ = self.env.reset()
        return self.discretize_state(state)
    
    def step(self, action):
        """
        تنفيذ خطوة في البيئة
        
        المكافآت (لا يمكن تعديلها):
        - كل خطوة: -1
        - الوصول للهدف: 0 (ولكن تنتهي الحلقة)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        return self.discretize_state(next_state), reward, done, info
    
    def render(self):
        """عرض البيئة"""
        return self.env.render()
    
    def close(self):
        """إغلاق البيئة"""
        self.env.close()


# ========================================================================
# 3️⃣ خوارزمية Q-Learning (يمكنك التعديل هنا!)
# ========================================================================

class QLearningAgent:
    """
    وكيل Q-Learning للتحدي
    
     يمكنك تعديل:
    - قيم المعاملات (learning_rate, discount_factor, etc.)
    - استراتيجية epsilon decay
    - طريقة اختيار الإجراء
    
     لا يمكنك:
    - استخدام Neural Networks
    - تغيير الخوارزمية الأساسية
    """
    
    def __init__(self, 
                 n_actions=3,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995):
        """
        المعاملات القابلة للتعديل:
        ---------------------------
        learning_rate: معدل التعلم (alpha) - جرب قيم بين 0.01 و 0.5
        discount_factor: معامل الخصم (gamma) - جرب قيم بين 0.9 و 0.999
        epsilon_start: قيمة epsilon الابتدائية
        epsilon_end: قيمة epsilon النهائية
        epsilon_decay: معدل تناقص epsilon
        """
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # جدول Q (يمكن استخدام defaultdict أو dict عادي)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
    
    def get_action(self, state, training=True):
        """
        اختيار إجراء باستخدام epsilon-greedy
        
        يمكنك تعديل هذه الدالة لتحسين الأداء!
        """
        if training and np.random.random() < self.epsilon:
            # استكشاف: اختيار عشوائي
            return np.random.randint(0, self.n_actions)
        else:
            # استغلال: اختيار أفضل إجراء
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        تحديث جدول Q
        
        صيغة Q-Learning:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # إذا انتهت الحلقة، لا يوجد حالة تالية
            max_next_q = 0
        else:
            # أقصى قيمة Q للحالة التالية
            max_next_q = np.max(self.q_table[next_state])
        
        # تحديث Q
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """تقليل epsilon بعد كل حلقة"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ========================================================================
# 4️⃣ دالة التدريب
# ========================================================================

def train_mountain_car(agent, env, n_episodes=1000, max_steps=200, verbose=True):
    """
    تدريب الوكيل على تحدي Mountain Car
    
    المعاملات:
    -----------
    agent: وكيل Q-Learning
    env: بيئة التحدي
    n_episodes: عدد الحلقات التدريبية
    max_steps: الحد الأقصى للخطوات في كل حلقة
    verbose: عرض التقدم
    
    المخرجات:
    ---------
    episode_rewards: قائمة بمكافآت كل حلقة
    episode_lengths: قائمة بأطوال كل حلقة
    """
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(" بدء التدريب...")
    print("=" * 70)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # اختيار وتنفيذ إجراء
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # تحديث Q-table
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                if step < max_steps - 1:  # نجح في الوصول
                    success_count += 1
                break
        
        # تقليل epsilon
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # عرض التقدم
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = (success_count / 100) * 100
            
            print(f"الحلقة {episode + 1:4d} | "
                  f"متوسط المكافأة: {avg_reward:7.2f} | "
                  f"متوسط الطول: {avg_length:5.1f} | "
                  f"معدل النجاح: {success_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            success_count = 0
    
    print("=" * 70)
    print(" اكتمل التدريب!")
    
    return episode_rewards, episode_lengths


# ========================================================================
# 5️⃣ دوال التصور والتقييم
# ========================================================================

def plot_training_results(episode_rewards, episode_lengths):
    """رسم نتائج التدريب"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(' نتائج التدريب - تحدي Mountain Car', 
                 fontsize=16, weight='bold')
    
    # 1. منحنى المكافآت
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='المكافأة')
    
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, color='red', linewidth=2, 
                label=f'المتوسط المتحرك ({window})')
    
    ax1.set_xlabel('رقم الحلقة', fontsize=11)
    ax1.set_ylabel('المكافأة الكلية', fontsize=11)
    ax1.set_title('منحنى التعلم - المكافآت', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. أطوال الحلقات
    ax2 = axes[0, 1]
    ax2.plot(episode_lengths, alpha=0.3, color='green', label='الطول')
    
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, 
                                np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), 
                moving_avg, color='orange', linewidth=2, 
                label=f'المتوسط المتحرك ({window})')
    
    ax2.set_xlabel('رقم الحلقة', fontsize=11)
    ax2.set_ylabel('عدد الخطوات', fontsize=11)
    ax2.set_title('طول الحلقات (أقل = أفضل)', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. التوزيع النهائي للمكافآت
    ax3 = axes[1, 0]
    last_100 = episode_rewards[-100:]
    ax3.hist(last_100, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(last_100), color='red', linestyle='--', 
                linewidth=2, label=f'المتوسط: {np.mean(last_100):.1f}')
    ax3.set_xlabel('المكافأة', fontsize=11)
    ax3.set_ylabel('التكرار', fontsize=11)
    ax3.set_title('توزيع المكافآت (آخر 100 حلقة)', fontsize=12, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. معدل النجاح
    ax4 = axes[1, 1]
    success_threshold = -200  # نجح إذا كانت المكافأة أكبر من -200
    success_rates = []
    
    for i in range(100, len(episode_rewards), 10):
        recent = episode_rewards[i-100:i]
        success_rate = (np.array(recent) > success_threshold).mean() * 100
        success_rates.append(success_rate)
    
    ax4.plot(range(100, len(episode_rewards), 10), success_rates, 
            color='teal', linewidth=2, marker='o', markersize=3)
    ax4.set_xlabel('رقم الحلقة', fontsize=11)
    ax4.set_ylabel('معدل النجاح (%)', fontsize=11)
    ax4.set_title('معدل النجاح (آخر 100 حلقة)', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.show()


def evaluate_agent(agent, env, n_episodes=100):
    """
    تقييم الوكيل المدرب
    
    المخرجات:
    ---------
    dict: إحصائيات الأداء
    """
    
    print("\n" + "=" * 70)
    print(" تقييم الأداء النهائي...")
    print("=" * 70)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(200):
            action = agent.get_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                if step < 199:
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
    
    # حساب الإحصائيات
    stats = {
        'متوسط_المكافأة': np.mean(episode_rewards),
        'انحراف_معياري_المكافأة': np.std(episode_rewards),
        'أفضل_مكافأة': np.max(episode_rewards),
        'أسوأ_مكافأة': np.min(episode_rewards),
        'متوسط_الخطوات': np.mean(episode_lengths),
        'معدل_النجاح_%': (success_count / n_episodes) * 100,
        'المجموع_النهائي': np.sum(episode_rewards)
    }
    
    # عرض النتائج
    print(f"\n النتائج على {n_episodes} حلقة:")
    print(f"   • متوسط المكافأة: {stats['متوسط_المكافأة']:.2f} ± {stats['انحراف_معياري_المكافأة']:.2f}")
    print(f"   • أفضل مكافأة: {stats['أفضل_مكافأة']:.2f}")
    print(f"   • أسوأ مكافأة: {stats['أسوأ_مكافأة']:.2f}")
    print(f"   • متوسط عدد الخطوات: {stats['متوسط_الخطوات']:.1f}")
    print(f"   • معدل النجاح: {stats['معدل_النجاح_%']:.1f}%")
    print(f"\n المجموع النهائي للنقاط: {stats['المجموع_النهائي']:.0f}")
    print("=" * 70)
    
    return stats


# ========================================================================
# 6️⃣ التشغيل الرئيسي
# ========================================================================

def main():
    """البرنامج الرئيسي للتحدي"""
    
    print("\n" + "=" * 70)
    print("  تحدي Mountain Car - المستوى الأول (مبتدئ)")
    print("=" * 70)
    
    # إنشاء البيئة
    env = MountainCarChallenge()
    
    # إعداد تقسيم الحالات (يمكنك تعديل هذه القيم!)
    env.setup_discretization(n_position_bins=20, n_velocity_bins=20)
    
    # إنشاء الوكيل (يمكنك تعديل المعاملات!)
    agent = QLearningAgent(
        n_actions=3,
        learning_rate=0.1,        # جرب: 0.05, 0.2, 0.5
        discount_factor=0.99,      # جرب: 0.95, 0.99, 0.999
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995        # جرب: 0.99, 0.999
    )
    
    print("\n⚙️  معاملات التعلم المستخدمة:")
    print(f"   • معدل التعلم (α): {agent.learning_rate}")
    print(f"   • معامل الخصم (γ): {agent.discount_factor}")
    print(f"   • Epsilon النهائي: {agent.epsilon_end}")
    print(f"   • معدل تناقص Epsilon: {agent.epsilon_decay}")
    print(f"   • تقسيم الحالات: 20×20")
    
    # التدريب
    episode_rewards, episode_lengths = train_mountain_car(
        agent, env, 
        n_episodes=1000,  # يمكنك زيادة العدد للتدريب الأطول
        max_steps=200,
        verbose=True
    )
    
    # رسم النتائج
    plot_training_results(episode_rewards, episode_lengths)
    
    # التقييم النهائي
    final_stats = evaluate_agent(agent, env, n_episodes=100)
    
    # إغلاق البيئة
    env.close()
    
    return agent, env, final_stats


# ========================================================================
#  تشغيل التحدي
# ========================================================================

if __name__ == "__main__":
    agent, env, stats = main()
    
    print("\n انتهى التحدي!")
    print(" نصيحة: جرب تعديل المعاملات في القسم 6️⃣ لتحسين الأداء")
    print(" لا تنسَ توثيق استراتيجيتك والتغييرات التي أجريتها!")
