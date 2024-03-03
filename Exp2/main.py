from train_model import Lab2Model

if __name__ == '__main__':
    model = Lab2Model(batch_size=64, num_workers=4, seed=0) #num_workers changed from 8 to 4

    # 最好还是使用 GPU, CPU 太慢了。。。
    model.train(lr=0.001, epochs=20, device='cuda', wait=4, lrd=True, fig_name='Final')

    # 选择好超参数后，测试模型表现
    print('Test score:', model.test())

