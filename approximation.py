import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # 用于GELU计算
from scipy.integrate import quad  # 用于误差计算
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import os

# Simulation of Approximation of non-linear functions using polynomial expansion

# Approximate the non-linear function using a polynomial expansion
class FunctionApproximator:
    def __init__(self, output_dir='results'):
        self.functions = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'GELU': lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2))),
            'GELU_minus_0.5x': lambda x: 0.5 * x * erf(x / np.sqrt(2)),
            'softmax': lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))),  # numerical stability
            'SiLU': lambda x: x / (1 + np.exp(-x)),  # Sigmoid Linear Unit
            "exp": lambda x: np.exp(x)
        }
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def polynomial(self, x, coeffs):
        return sum(c * (x**i) for i, c in enumerate(coeffs))
    
    def compare_functions(self, func_name, degree, coeffs, x_range=(-5, 5), error_range=None):
        error_range = error_range or x_range
        x = np.linspace(x_range[0], x_range[1], 1000)
        target_func = self.functions[func_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, target_func(x), label=f'True {func_name}')
        plt.plot(x, self.polynomial(x, coeffs), '--', label=f'Approx (deg={degree})')
        
        # 计算标准化误差
        abs_error = lambda x: abs(target_func(x) - self.polynomial(x, coeffs))
        avg_error, _ = quad(abs_error, *error_range)
        avg_error /= (error_range[1] - error_range[0])
        
        plt.title(f'{func_name} Approximation\nAvg Error: {avg_error:.2e}')
        plt.legend()
        plt.grid()
        
        # 保存图像
        filename = f"{func_name}_deg{degree}_{'_'.join(map(str,coeffs))}.png".replace('.','p')
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return avg_error, save_path
    
    def compare_exp(self, func_name, degree, x_range=(-10, 0), error_range=None):
        error_range = error_range or x_range
        x = np.linspace(x_range[0], x_range[1], 1000)
        target_func = self.functions[func_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, target_func(x), label=f'True {func_name}')
        plt.plot(x, (1+ x/ (2**degree))**(2**degree), '--', label=f'Approx (deg={degree})')
        
        # 计算标准化误差
        abs_error = lambda x: abs(target_func(x) - (1+ x/ (2**degree))**(2**degree))
        avg_error, _ = quad(abs_error, *error_range)
        avg_error /= (error_range[1] - error_range[0])
        
        plt.title(f'{func_name} Approximation\nAvg Error: {avg_error:.2e}')
        plt.legend()
        plt.grid()
        
        # 保存图像
        filename = f"{func_name}_deg{degree}.png".replace('.','p')
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return avg_error, save_path
    
    def remez_approximation(self, func_name, degree, interval, max_iter=1000, tol=1e-3):
        a, b = interval
        n = degree + 2
        target_func = self.functions[func_name]
       
        # 改进初始点：非对称切比雪夫节点
        k = np.arange(1, n+1)
        nodes = (a + b)/2 - (b - a)/2 * np.cos((2*k-1)*np.pi/(2*n))  # 非对称调整

        # nodes = np.cos(np.linspace(0, np.pi, n)) * (b-a)/2 + (a+b)/2
        
        for _ in range(max_iter):
            V = np.vander(nodes, degree+1, increasing=True)
            coeffs = np.linalg.lstsq(V, target_func(nodes), rcond=None)[0]
            
            # 寻找最大误差点
            def error_func(x):
                approx = np.polyval(coeffs[::-1], x)
                return np.abs(target_func(x) - approx)
        
            # 在区间内寻找最大误差点
            max_err_points = []
            for _ in range(degree+2):
                res = minimize_scalar(lambda x: -error_func(x), bounds=(a,b), method='bounded')
                max_err_points.append(res.x)
            
            # sample_points = np.linspace(a, b, num=1000)  # 采样密度可调整
            # mean_err = np.mean(error_func(sample_points))
            
            # 检查收敛
            new_nodes = np.array(sorted(max_err_points))
            if np.mean(np.abs(new_nodes - nodes)) < tol:
                break
            nodes = new_nodes

            # # 全局误差分析（关键改进）
            # x_test = np.linspace(a, b, 1000)
            # errors = np.abs(target_func(x_test) - np.polyval(coeffs[::-1], x_test))
            # max_err_idx = np.argmax(errors)
            # candidate = x_test[max_err_idx]
            
            # # 更新节点集
            # new_nodes = np.sort(np.append(nodes[1:-1], candidate))
            # new_nodes[0], new_nodes[-1] = a, b  # 固定端点
        
        return coeffs
    
        def remez_approximation_plus(self, func_name, degree, interval, max_iter=500, tol=1e-3):
            a, b = interval
            n = degree + 2
            target_func = self.functions[func_name]
           
            # 改进初始点：非对称切比雪夫节点
            k = np.arange(1, n+1)
            nodes = (a + b)/2 - (b - a)/2 * np.cos((2*k-1)*np.pi/(2*n))  # 非对称调整
            
            for _ in range(max_iter):
                V = np.vander(nodes, degree+1, increasing=True)
                coeffs = np.linalg.lstsq(V, target_func(nodes), rcond=None)[0]
                
                # 寻找最大误差点
                def error_func(x):
                    approx = np.polyval(coeffs[::-1], x)
                    return np.abs(target_func(x) - approx)
            
                # 在区间内寻找最大误差点
                max_err_points = []
                for _ in range(degree+2):
                    res = minimize_scalar(lambda x: -error_func(x), bounds=(a,b), method='bounded')
                    max_err_points.append(res.x)
                
                # 检查收敛
                new_nodes = np.array(sorted(max_err_points))
                if np.max(np.abs(new_nodes - nodes)) < tol:
                    break
                nodes = new_nodes

                # # 全局误差分析（关键改进）
                # x_test = np.linspace(a, b, 1000)
                # errors = np.abs(target_func(x_test) - np.polyval(coeffs[::-1], x_test))
                # max_err_idx = np.argmax(errors)
                # candidate = x_test[max_err_idx]
                
                # # 更新节点集
                # new_nodes = np.sort(np.append(nodes[1:-1], candidate))
                # new_nodes[0], new_nodes[-1] = a, b  # 固定端点
            
            return coeffs


    # 非Remez方法：多项式拟合GELU-0.5x
    # 通过最小化平均绝对误差（MAE）来拟合多项式
    # 修改：不应该是最小绝对误差 应该是最小化误差的百分比（相对误差）
    # 这里的GELU-0.5x是指GELU函数减去0.5x的部分

    def poly_func(self, coeffs, x):
        """多项式计算函数"""
        return sum(c * x**i for i, c in enumerate(coeffs))

    def mae_loss(self, coeffs, x_samples):
        """平均绝对误差计算"""
        target_func = self.functions["GELU"]
        approx = self.poly_func(coeffs, x_samples)
        return np.mean(np.abs(target_func(x_samples) - approx))

    def fit_gelu_poly(self, degree=3, x_range=(0, 2.7), n_samples=5000):
        # 生成训练样本
        x_samples = np.linspace(x_range[0], x_range[1], n_samples)
        
        # 初始猜测（奇数项系数设为0）
        # need to be optimized
        init_coeffs = [0.1 if i % 2 == 0 else 0 for i in range(degree+1)]
        
        # 优化MAE
        result = minimize(self.mae_loss, init_coeffs, args=(x_samples,),
                        method='L-BFGS-B')
        
        # 整理系数（低次项在前）
        return result.x




if __name__ == "__main__":
    approximator = FunctionApproximator()
    # 示例测试
    # sigmoid_coeffs = [0.5, 0.15, -0.001, -0.003]
    # error, path = approximator.compare_functions('sigmoid', 3, sigmoid_coeffs)
    
    # 示例1：用3次多项式近似sigmoid (系数需自行调整)
    # coeffs = [0.5, 0.15, -0.001, -0.003]  # 常数项、1次、2次、3次系数
    # error0, path0 = approximator.compare_functions('sigmoid', degree=3, coeffs=coeffs, x_range=(-5,5))
    # print(f"sigmoid近似结果的平均误差: {error0:.2e}")
    # print(f"结果已保存至: {path0}")
    
    # # 示例2：用5次多项式近似GELU
    # gelu_coeffs = [0, 0.5, 0.1, -0.01, 0.002, -0.0002]
    # error1, path1 = approximator.compare_functions('GELU', degree=5, coeffs=gelu_coeffs, x_range=(-4,4))
    # print(f"GELU近似结果的平均误差: {error1:.2e}")
    # print(f"结果已保存至: {path1}")
    
    # # 示例3：用4次多项式近似softmax (注意softmax是向量函数，这里简化为单变量)
    # softmax_coeffs = [0.1, 0.05, -0.002, 0.0001]  # 常数项、1次、2次、3次系数
    # error2, path2 = approximator.compare_functions('softmax', degree=4, coeffs=softmax_coeffs, x_range=(-5,5))
    # print(f"softmax近似结果的平均误差: {error2:.2e}")
    # print(f"结果已保存至: {path2}")


    # BOLT Approximation
    # gelu_coeffs_bolt = [0.001620808531841547, 0.03798164612714154+0.5, 0.5410550166368381, 0.18352506127082727, 0.020848611754127593]
    # error_bolt, path_bolt = approximator.compare_functions('GELU', degree=4, coeffs=gelu_coeffs_bolt, x_range=(-2.7,0))
    # print(f"GELU_BOLT近似结果的平均误差: {error_bolt:.2e}")
    # print(f"结果已保存至: {path_bolt}")

    # 计算1-4次多项式近似 (GELU-0.5x) Remez算法
    # approximations = {}
    # for degree in range(1, 3):
    #     coeffs = approximator.remez_approximation(degree=degree,func_name="GELU_minus_0.5x", interval=[0, 2.7], tol=1e-3)
    #     approximations[degree] = coeffs
    #     print(f"{degree}次多项式系数: {coeffs}")
    
    # for degree in range(3, 5):
    #     coeffs = approximator.remez_approximation(degree=degree,func_name="GELU_minus_0.5x", interval=[0, 2.7], tol=1e-4)
    #     approximations[degree] = coeffs
    #     print(f"{degree}次多项式系数: {coeffs}")

    # error_1, path_1 = approximator.compare_functions('GELU', degree=1, coeffs=approximations[1]+[0,0.5], x_range=(0,2.7))
    # error_2, path_2 = approximator.compare_functions('GELU', degree=2, coeffs=approximations[2]+[0,0.5,0], x_range=(0,2.7))
    # error_3, path_3 = approximator.compare_functions('GELU', degree=3, coeffs=approximations[3]+[0,0.5,0,0], x_range=(0,2.7))
    # error_4, path_4 = approximator.compare_functions('GELU', degree=4, coeffs=approximations[4]+[0,0.5,0,0,0], x_range=(0,2.7))
   
    # print(f"GELU 1次多项式近似结果的平均误差: {error_1:.2e}")
    # print(f"GELU 2次多项式近似结果的平均误差: {error_2:.2e}")
    # print(f"GELU 3次多项式近似结果的平均误差: {error_3:.2e}")
    # print(f"GELU 4次多项式近似结果的平均误差: {error_4:.2e}")
    # print(f"1次多项式结果已保存至: {path_1}")
    # print(f"2次多项式结果已保存至: {path_2}")
    # print(f"3次多项式结果已保存至: {path_3}")
    # print(f"4次多项式结果已保存至: {path_4}")
    

    # (0, 2.7)区间的GELU多项式拟合
    coeffs_fit = {}
    error_fit = []
    path_fit = []
    
    for i in range(0,2):
        coeffs_tmp = approximator.fit_gelu_poly(degree=i+1, x_range=(-0.75, 2.7))
        coeffs_fit[i+1] = coeffs_tmp
        print(f"{i+1}次多项式拟合系数: {coeffs_tmp}")
        error_tmp, path_tmp = approximator.compare_functions('GELU', degree=i+1, coeffs=coeffs_tmp, x_range=(-0.75, 2.7))
        error_fit.append(error_tmp)
        path_fit.append(path_tmp)

    # for i in range(7):
    #     print(f"{i+1}次多项式拟合指数函数")
    #     error_tmp, path_tmp = approximator.compare_exp(func_name='exp', degree=i+1, x_range=(-10, 0))
    #     error_fit.append(error_tmp)
    #     path_fit.append(path_tmp)
    #     print(f"{i+1}次多项式拟合指数函数的平均误差: {error_tmp:.2e}")
        
    
    # 计算GELU的多项式近似
    # 
    # error_fit_2, path_fit_2 = approximator.compare_functions('GELU', degree=2, coeffs=coeffs_fit_2+[0,0.5,0], x_range=(0, 2.7))
    # error_fit_3, path_fit_3 =  approximator.compare_functions('GELU', degree=3, coeffs=coeffs_fit_3+[0,0.5,0,0], x_range=(0, 2.7))
    # error_fit_4, path_fit_4 = approximator.compare_functions('GELU', degree=4, coeffs=coeffs_fit_4+[0,0.5,0,0,0], x_range=(0, 2.7))



    # (-2.7, 0)区间的GELU多项式拟合
    # coeffs_fit_1 = approximator.fit_gelu_poly(degree=1, x_range=(-2.7, 0))
    # coeffs_fit_2 = approximator.fit_gelu_poly(degree=2, x_range=(-2.7, 0))
    # coeffs_fit_3 = approximator.fit_gelu_poly(degree=3, x_range=(-2.7, 0))
    # coeffs_fit_4 = approximator.fit_gelu_poly(degree=4, x_range=(-2.7, 0))

        # error_fit_1, path_fit_1 = approximator.compare_functions('GELU', degree=1, coeffs=coeffs_fit_1+[0,0.5], x_range=(-2.7, 0))
    # error_fit_2, path_fit_2 = approximator.compare_functions('GELU', degree=2, coeffs=coeffs_fit_2+[0,0.5,0], x_range=(-2.7, 0))
    # error_fit_3, path_fit_3 =  approximator.compare_functions('GELU', degree=3, coeffs=coeffs_fit_3+[0,0.5,0,0], x_range=(-2.7, 0))
    # error_fit_4, path_fit_4 = approximator.compare_functions('GELU', degree=4, coeffs=coeffs_fit_4+[0,0.5,0,0,0], x_range=(-2.7, 0))
    
    # print(f"拟合1次多项式近似结果的平均误差: {error_fit_1:.2e}")
    # print(f"拟合2次多项式近似结果的平均误差: {error_fit_2:.2e}")
    # print(f"拟合3次多项式近似结果的平均误差: {error_fit_3:.2e}")
    # print(f"拟合4次多项式近似结果的平均误差: {error_fit_4:.2e}")