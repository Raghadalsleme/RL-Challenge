#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========================================================================
# تحدي التعلم الذاتي (للمتخصصين)
# المستوى الخامس: التوازن الفائق (Impossible CartPole)
# الصعوبة: فائقة الصعوبة
# 
# اسم الملف: impossible_cartpole_challenge.py
# تاريخ الإنشاء: 2025
# ========================================================================

"""
 وصف التحدي:
--------------
نسخة معدّلة وشديدة الصعوبة من CartPole الكلاسيكي.
عربة مع عمود يجب موازنته في ظروف قاسية:
- عمود أطول وأثقل
- حركة أسرع
- هوامش خطأ أضيق
- رياح عشوائية ومفاجئة
- احتكاك غير منتظم

 القوانين والقيود:
-------------------
1. يجب استخدام خوارزمية Q-Learning فقط
2. لا يسمح باستخدام Deep Learning أو Neural Networks
3. الحالة: 4 أبعاد (موقع، سرعة، زاوية، سرعة زاوية)
4. الإجراءات: يسار (0) أو يمين (1)
5. النجاح = البقاء متوازناً لأطول فترة ممكنة

التعديلات الصعبة:
- طول العمود: 2× الطول العادي
- كتلة العمود: 3× الكتلة العادية
- قوة الدفع: 50% من القوة العادية
- زاوية السقوط: ±8° بدلاً من ±12°
- رياح عشوائية كل 10-30 خطوة
- احتكاك متغير

 معايير التقييم:
------------------
- الهدف: البقاء لأكثر من 500 خطوة (صعب جداً)
- المتوسط الجيد: > 200 خطوة
- المتوسط العادي: 50-100 خطوة
- المجموع النهائي: متوسط آخر 100 حلقة

 تنبيهات هامة:
-----------------
- هذا التحدي مستحيل تقريباً
- صُمم ليكون على حافة قدرات Q-Learning
- يحتاج ضبط دقيق جداً للمعاملات
- استراتيجيات تبسيط الحالة حرجة
- التوقعات: النجاح الجزئي فقط
"""

# ========================================================================
# 1️⃣ استيراد المكتبات المطلوبة
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 2️⃣ إعداد البيئة (لا تعدل هذا القسم!)
# ========================================================================

class ImpossibleCartPoleEnv(gym.Env):
    """
    نسخة مستحيلة من CartPole
     ممنوع التعديل على هذا الكلاس!
    
    التعديلات الصعبة:
    - عمود أطول وأثقل (صعب الموازنة)
    - قوة دفع أقل (استجابة بطيئة)
    - زوايا سقوط أضيق
    - رياح عشوائية
    - احتكاك متغير
    """
    
    def __init__(self):
        super(ImpossibleCartPoleEnv, self).__init__()
        
        # معاملات فيزيائية صعبة
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.3  # 3× الوزن العادي!
        self.total_mass = self.masspole + self.masscart
        self.length = 1.0  # 2× الطول العادي!
        self.polemass_length = self.masspole * self.length
        self.force_mag = 5.0  # 50% من القوة العادية!
        self.tau = 0.02
        
        # حدود أضيق
        self.theta_threshold_radians = 8 * 2 * np.pi / 360  # ±8° فقط!
        self.x_threshold = 2.4
        
        # رياح عشوائية
        self.wind_force = 0
        self.wind_counter = 0
        self.wind_interval = np.random.randint(10, 30)
        
        # احتكاك متغير
        self.friction = 0.1
        
        # حدود الحالة
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.state = None
        self.steps_beyond_done = None
    
    def reset(self, seed=None, options=None):
        """إعادة تعيين البيئة"""
        super().reset(seed=seed)
        
        # حالة ابتدائية عشوائية (أصعب)
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        self.steps_beyond_done = None
        self.wind_force = 0
        self.wind_counter = 0
        self.wind_interval = np.random.randint(10, 30)
        
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        """تنفيذ خطوة في البيئة"""
        assert self.action_space.contains(action)
        
        x, x_dot, theta, theta_dot = self.state
        
        # القوة المطبقة
        force = self.force_mag if action == 1 else -self.force_mag
        
        # رياح عشوائية
        self.wind_counter += 1
        if self.wind_counter >= self.wind_interval:
            self.wind_force = np.random.uniform(-3, 3)
            self.wind_counter = 0
            self.wind_interval = np.random.randint(10, 30)
        
        force += self.wind_force
        
        # احتكاك متغير
        self.friction = 0.05 + 0.1 * np.random.random()
        force -= self.friction * x_dot
        
        # حساب الفيزياء
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # تحديث الحالة
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        # شروط الفشل الصارمة
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        
        return np.array(self.state, dtype=np.float32), reward, done, False, {}


class ImpossibleCartPoleChallenge:
    """
    واجهة تحدي Impossible CartPole
     ممنوع التعديل!
    """
    
    def __init__(self):
        self.env = ImpossibleCartPoleEnv()
        self.state_bins = None
    
    def setup_discretization(self, bins_per_dimension=20):
        """
        إعداد تقسيم الحالات
        
        المعاملات:
        -----------
        bins_per_dimension: عدد الأقسام لكل بُعد
        
         نصيحة: القيم الأعلى = دقة أكثر لكن تدريب أبطأ
        """
        # حدود دقيقة لكل بُعد
        self.state_bounds = [
            (-2.4, 2.4),      # x position
            (-3.0, 3.0),      # x velocity
            (-0.21, 0.21),    # theta (±12°)
            (-3.0, 3.0)       # theta velocity
        ]
        
        self.state_bins = []
        for low, high in self.state_bounds:
            self.state_bins.append(
                np.linspace(low, high, bins_per_dimension)
            )
    
    def discretize_state(self, state):
        """تحويل الحالة المستمرة إلى منفصلة"""
        discrete_state = []
        
        for i, (s, bins) in enumerate(zip(state, self.state_bins)):
            s_clipped = np.clip(s, self.state_bounds[i][0], 
                               self.state_bounds[i][1])
            idx = np.digitize(s_clipped, bins)
            discrete_state.append(idx)
        
        return tuple(discrete_state)
    
    def reset(self):
        """إعادة تعيين البيئة"""
        state, _ = self.env.reset()
        return self.discretize_state(state)
    
    def step(self, action):
        """تنفيذ خطوة"""
        next_state, reward, done, truncated, info = self.env.step(action)
        return self.discretize_state(next_state), reward, done or truncated, info
    
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
    وكيل Q-Learning للتحدي المستحيل
    
     يمكنك تعديل:
    - جميع المعاملات
    - استراتيجيات متقدمة
    - أساليب التعلم
    
     نصائح للنجاح:
    - learning_rate عالي في البداية
    - discount_factor قريب من 1.0
    - epsilon_decay بطيء
    - جرب Optimistic Initialization
    """
    
    def __init__(self, 
                 n_actions=2,
                 learning_rate=0.5,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9995,
                 optimistic_init=0.0):
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        
        # جدول Q مع قيم تفاؤلية اختيارية
        self.q_table = defaultdict(
            lambda: np.ones(n_actions) * optimistic_init
        )
        
        self.training_episodes = 0
    
    def get_action(self, state, training=True):
        """اختيار إجراء"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """تحديث جدول Q"""
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        target_q = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """تقليل epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1


# ========================================================================
# 4️⃣ دالة التدريب
# ========================================================================

def train_impossible_cartpole(agent, env, n_episodes=5000, verbose=True):
    """
    تدريب الوكيل على التحدي المستحيل
    
     تحذير: يحتاج تدريب مكثف
    """
    
    episode_rewards = []
    episode_lengths = []
    
    print(" بدء التدريب على  CartPole...")
    print("=" * 70)
    print(" التحدي الأصعب  !")
    print("   • عمود أطول وأثقل")
    print("   • قوة دفع أقل")
    print("   • رياح عشوائية")
    print("   • هوامش خطأ ضيقة جداً")
    print("=" * 70)
    
    best_score = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode_reward > best_score:
            best_score = episode_reward
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            status = "🔴"
            if avg_reward > 200:
                status = "🟢 ممتاز!"
            elif avg_reward > 100:
                status = "🟡 جيد"
            
            print(f"الحلقة {episode + 1:5d} | "
                  f"متوسط: {avg_reward:6.1f} | "
                  f"أفضل: {best_score:6.0f} | "
                  f"Epsilon: {agent.epsilon:.3f} {status}")
    
    print("=" * 70)
    print(" اكتمل التدريب!")
    print(f"   أفضل أداء: {best_score:.0f} خطوة")
    
    return episode_rewards, episode_lengths


# ========================================================================
# 5️⃣ دوال التصور والتقييم
# ========================================================================

def plot_training_results(episode_rewards, episode_lengths):
    """رسم نتائج التدريب"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 نتائج التدريب - Impossible CartPole', 
                 fontsize=16, weight='bold')
    
    # منحنى المكافآت
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.2, color='blue')
    
    # خطوط الأهداف
    ax1.axhline(y=500, color='gold', linestyle='--', linewidth=2, 
                alpha=0.7, label='مستحيل (500)')
    ax1.axhline(y=200, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='ممتاز (200)')
    ax1.axhline(y=100, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='جيد (100)')
    
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, color='red', linewidth=3, label='المتوسط')
    
    ax1.set_xlabel('رقم الحلقة')
    ax1.set_ylabel('عدد الخطوات')
    ax1.set_title('منحنى التعلم')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # أطوال الحلقات (نفس الشيء)
    ax2 = axes[0, 1]
    ax2.plot(episode_lengths, alpha=0.2, color='green')
    
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, 
                                np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), 
                moving_avg, color='orange', linewidth=3)
    
    ax2.set_xlabel('رقم الحلقة')
    ax2.set_ylabel('عدد الخطوات')
    ax2.set_title('استقرار الأداء')
    ax2.grid(True, alpha=0.3)
    
    # توزيع الأداء
    ax3 = axes[1, 0]
    last_500 = episode_rewards[-500:]
    ax3.hist(last_500, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(last_500), color='red', linestyle='--', 
                linewidth=3, label=f'المتوسط: {np.mean(last_500):.1f}')
    ax3.axvline(np.median(last_500), color='blue', linestyle='--', 
                linewidth=2, label=f'الوسيط: {np.median(last_500):.1f}')
    ax3.set_xlabel('عدد الخطوات')
    ax3.set_ylabel('التكرار')
    ax3.set_title('توزيع الأداء (آخر 500 حلقة)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # الأرقام القياسية
    ax4 = axes[1, 1]
    records = []
    current_best = 0
    for reward in episode_rewards:
        if reward > current_best:
            current_best = reward
        records.append(current_best)
    
    ax4.plot(records, color='gold', linewidth=3)
    ax4.fill_between(range(len(records)), records, alpha=0.3, color='gold')
    ax4.set_xlabel('رقم الحلقة')
    ax4.set_ylabel('الرقم القياسي')
    ax4.set_title('تطور الرقم القياسي')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_agent(agent, env, n_episodes=100):
    """تقييم الوكيل المدرب"""
    
    print("\n" + "=" * 70)
    print("📈 التقييم النهائي...")
    print("=" * 70)
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.get_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    stats = {
        'متوسط_الخطوات': np.mean(episode_rewards),
        'انحراف_معياري': np.std(episode_rewards),
        'أفضل_أداء': np.max(episode_rewards),
        'أسوأ_أداء': np.min(episode_rewards),
        'الوسيط': np.median(episode_rewards),
        'المجموع_النهائي': np.sum(episode_rewards)
    }
    
    print(f"\n النتائج على {n_episodes} حلقة:")
    print(f"   • متوسط الخطوات: {stats['متوسط_الخطوات']:.1f} ± {stats['انحراف_معياري']:.1f}")
    print(f"   • الوسيط: {stats['الوسيط']:.1f}")
    print(f"   • أفضل أداء: {stats['أفضل_أداء']:.0f}")
    print(f"   • أسوأ أداء: {stats['أسوأ_أداء']:.0f}")
    print(f"\n المجموع النهائي: {stats['المجموع_النهائي']:.0f}")
    
    # تقييم المستوى
    avg = stats['متوسط_الخطوات']
    print("\n  التقييم:")
    if avg >= 500:
        print("   ⭐⭐⭐⭐⭐ مستحيل! حققت المستحيل!")
    elif avg >= 300:
        print("   ⭐⭐⭐⭐ استثنائي! أداء خارق!")
    elif avg >= 200:
        print("   ⭐⭐⭐ ممتاز! نتيجة رائعة!")
    elif avg >= 100:
        print("   ⭐⭐ جيد جداً! أداء قوي")
    elif avg >= 50:
        print("   ⭐ مقبول - يمكن التحسين")
    else:
        print("   💪 استمر في التدريب!")
    
    print("=" * 70)
    
    return stats


# ========================================================================
# 6️⃣ التشغيل الرئيسي
# ========================================================================

def main():
    """البرنامج الرئيسي للتحدي"""
    
    print("\n" + "=" * 70)
    print(" Impossible CartPole - المستوى الخامس (فائق الصعوبة)")
    print("=" * 70)
    
    # إنشاء البيئة
    env = ImpossibleCartPoleChallenge()
    env.setup_discretization(bins_per_dimension=20)
    
    # إنشاء الوكيل بمعاملات محسّنة
    agent = QLearningAgent(
        n_actions=2,
        learning_rate=0.3,          # جرب: 0.2-0.5
        discount_factor=0.99,        # جرب: 0.95-0.999
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,        # جرب: 0.995-0.9999
        optimistic_init=0.0          # جرب: 0, 5, 10
    )
    
    print("\n⚙️  معاملات التعلم:")
    print(f"   • معدل التعلم: {agent.learning_rate}")
    print(f"   • معامل الخصم: {agent.discount_factor}")
    print(f"   • Epsilon decay: {agent.epsilon_decay}")
    print(f"   • Optimistic init: {agent.optimistic_init}")
    
    print("\n نصائح للفوز:")
    print("   - زد عدد الحلقات (5000-10000)")
    print("   - جرب learning_rate أعلى (0.3-0.5)")
    print("   - جرب epsilon_decay أبطأ (0.999)")
    print("   - جرب optimistic initialization")
    
    # التدريب المكثف
    episode_rewards, episode_lengths = train_impossible_cartpole(
        agent, env, 
        n_episodes=5000,
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
# 🚀 تشغيل التحدي
# ========================================================================

if __name__ == "__main__":
    agent, env, stats = main()
    
    print("\n انتهى التحدي !")
    print("\n ماذا تعلمنا:")
    print("   - Q-Learning قوي لكن له حدود")
    print("   - الضبط الدقيق للمعاملات حاسم")
    print("   - التبسيط الذكي مفتاح النجاح")
    print("   - الصبر والتجريب ضروريان")
    print("\n تهانينا على إكمال جميع التحديات!")
