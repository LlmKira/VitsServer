import torch

import commons
import models


class SynthesizerTrn(models.SynthesizerTrn):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):

        super().__init__(
            n_vocab,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            n_speakers=n_speakers,
            gin_channels=gin_channels,
            use_sdp=use_sdp,
            **kwargs
        )

    def infer(self, x, x_lengths,
              sid=None,
              noise_scale=1,
              length_scale=1,
              noise_scale_w=1.,
              max_len=None,
              emotion_embedding=None):
        from ONNXVITS_utils import runonnx

        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        x, m_p, logs_p, x_mask = runonnx("ONNX_net/enc_p.onnx", x=x.numpy(), x_lengths=x_lengths.numpy())
        x = torch.from_numpy(x)
        m_p = torch.from_numpy(m_p)
        logs_p = torch.from_numpy(logs_p)
        x_mask = torch.from_numpy(x_mask)
        zinput = (torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale_w)
        if sid is not None:
            g = runonnx("ONNX_net/dp.onnx", sid=sid.numpy())
            g = torch.from_numpy(g).unsqueeze(-1)
            logw = runonnx("ONNX_net/dp.onnx", x=x.numpy(), x_mask=x_mask.numpy(), zin=zinput.numpy(), g=g.numpy())
        else:
            g = None
            logw = runonnx("ONNX_net/dp.onnx", x=x.numpy(), x_mask=x_mask.numpy(), zin=zinput.numpy())

        # logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)

        logw = torch.from_numpy(logw[0])

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1,
                                                                                 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # z = self.flow(z_p, y_mask, g=g, reverse=True)
        if sid is not None:
            z = runonnx("ONNX_net/flow.onnx", z_p=z_p.numpy(), y_mask=y_mask.numpy(), g=g.numpy())
            z = torch.from_numpy(z[0])
            o = runonnx("ONNX_net/dec.onnx", z_in=(z * y_mask)[:, :, :max_len].numpy(), g=g.numpy())
            o = torch.from_numpy(o[0])
        else:
            z = runonnx("ONNX_net/flow.onnx", z_p=z_p.numpy(), y_mask=y_mask.numpy())
            z = torch.from_numpy(z[0])
            o = runonnx("ONNX_net/dec.onnx", z_in=(z * y_mask)[:, :, :max_len].numpy())
            o = torch.from_numpy(o[0])
        # o = self.dec((z * y_mask)[:,:,:max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
